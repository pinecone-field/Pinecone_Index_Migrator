import json
import logging
import os
import random
import time
import traceback
from datetime import datetime
from itertools import islice
import pandas as pd
from pinecone.grpc import PineconeGRPC as Pinecone
import multiprocessing
import threading
import unicodedata
import re
import dbm
from tqdm import tqdm
import uuid
from concurrent.futures import ProcessPoolExecutor

# --- Load Config.json and set globals ---
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("Missing config.json file")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

API_KEY = config["api_key"]

PRIME_INDEX_STORE = config["prime_index_store"]

#for migrating across indexes
TARGET_API_KEY = config["api_key_dest"]
USE_2ND_TARGET_CLIENT = config["use_seperate_target_client"]

SOURCE_INDEX_NAME = config["source_index"]
TARGET_INDEX_NAME = config["target_index"]

CLOUD = config.get("target_cloud")
REGION = config.get("target_region")
SOURCE_NAMESPACE = config.get("source_namespace", "")
TARGET_NAMESPACE = config.get("target_namespace", "")
BATCH_SIZE = config.get("batch_size", 100)
#MAX_PAYLOAD_BYTES = 2 * 1024 * 1024  # 2MB limit
MAX_PAYLOAD_BYTES = 4194304
DELTA_MODE = config.get("delta_mode", False)
RESET_MIGRATION_FLAGS = config.get("reset_migration_flags", False)
VERIFY_ONLY = config.get("verify_only", False)
RUN_VALIDATION_AFTER_BATCH = config.get("run_validation_after_every_batch")
WRITE_TO_PARQUET = config.get("write_to_parquet")
USE_FILTER = config.get("use_filter")
FILTER_TO_USE = config.get("filter_to_use")
CREATE_TARGET = config.get("create_target")
PARQUET_WRITE_BATCH_SIZE = 10000  # or whatever size you want for Parquet chunks
PARQUET_WRITE_BACK_BATCH_SIZE = config.get("parquet_write_back_batch_size")
NAMESPACE_FROM_METADATA_FIELD = config.get("namespace_from_metadata_field")
METADATA_FIELD_TO_NAMESPACE_FROM = config.get("metadata_field_to_namespace_from")
ITERATE_OVER_NAMESPACES = config.get("iterate_over_namespaces")
USE_SOURCE_AS_TARGET_NAMESPACE = config.get("use_source_as_target_namespace")
DELTA_DELAY_BETWEEN_RUNS = config.get("delta_delay_between_runs")
DELTA_DELAY_DURATION = config.get("delta_delay_duration")

SAVE_TO_S3 = config.get("save_to_s3")
S3_BUCKET_NAME = config.get("bucket_name")
S3_FOLDER_NAME = config.get("s3_folder_name")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Pinecone Client ---
pc = Pinecone(api_key=API_KEY)
source_index = pc.Index(SOURCE_INDEX_NAME)

if USE_2ND_TARGET_CLIENT:
    pc_target = Pinecone(api_key=TARGET_API_KEY)
else:
    pc_target = pc


if PRIME_INDEX_STORE:
    print("Note: This operation takes a long time depending on number of vectors. Example: Expect like 30 minutes to an hour for like 5 million vectors.")
    print(f"Number of vectors: {source_index.describe_index_stats()['total_vector_count']}")
    id_db = dbm.open('index_list', 'c')

    cnt = 0
    for page in source_index.list(namespace=SOURCE_NAMESPACE):
        for vid in page:
            id_db[vid] = b"false"
            cnt = cnt + 1
            if ((cnt % 10000) == 0):
                logging.info(f"Fetched: {cnt}")

    exit(0)

# for using multiple processors
#MAX_WORKERS = min(32, (multiprocessing.cpu_count() or 1) * 2)
MAX_WORKERS = (min(16, (multiprocessing.cpu_count() or 1))) * 2

# handle keyboard input to early quit
stop_requested = False
stop_event = threading.Event()

#If iterate over namespace flag is true, use this to set if we need to move to the next namespace when no vectors are found
next_namespace_flag = False

#############################################################
#  Methods
#############################################################

_worker = {}

def init_worker():
    _worker["pc_src"] = Pinecone(api_key=API_KEY)
    _worker["src_index"] = _worker["pc_src"].Index(SOURCE_INDEX_NAME)

    _worker["pc_tgt"] = Pinecone(api_key=(TARGET_API_KEY if USE_2ND_TARGET_CLIENT else API_KEY))
    _worker["tgt_index"] = _worker["pc_tgt"].Index(TARGET_INDEX_NAME)


"""
    Used to clean non-ascii printable characters in namespace generation from meta-data field
"""
def sanitize_namespace(name):
    # Normalize to closest ASCII equivalent
    normalized = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    # Replace spaces and other non-word characters with underscore
    return re.sub(r'\W+', '_', normalized)

"""
Feature: Stream vectors to Parquet with metadata update
Description:
   - Writes the current batch of vectors to a uniquely named Parquet file.
   - Updates the source index to set `migrated` and `migrated_at` flags.
Parameters:
   - vectors: Dictionary of vector ID -> Vector object
   - output_dir: Directory to write Parquet files
   - source_index: Pinecone Index object to update metadata
   - namespace: Pinecone namespace for source vectors
   - batch_num: Integer used to differentiate output files
"""
def write_vectors_to_parquet(vectors, output_dir, source_index, namespace):
    if output_dir == "":
        output_dir = "default-directory"
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for vid, vec in vectors.items():
        metadata = vec.metadata or {}

        record = {
            "id": vid,
            "values": vec.values,
            "metadata": json.dumps(metadata)
        }

        records.append(record)

    if (SAVE_TO_S3):
        try:
            filename = f"s3://{S3_BUCKET_NAME}/{S3_FOLDER_NAME}/part_{uuid.uuid4().hex}.parquet"
            pd.DataFrame(records, columns=["id", "values", "metadata"]).to_parquet(filename, engine="pyarrow", index=False, compression='None')
            logging.info(f"Write Parquet File: {filename}")
        except Exception as e:
            logging.error(e)
    else:
        df = pd.DataFrame(records, columns=["id", "values", "metadata"])

        unique_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        parquet_path = os.path.join(output_dir, f"vectors_{unique_str}.parquet")

        df.to_parquet(parquet_path, index=False)
        logging.info(f"Wrote {len(records)} vectors to {parquet_path}")

"""
Helper to keep vector query from returning nothing
"""
def random_vector(dimension):
    return [random.uniform(-1.0, 1.0) for _ in range(dimension)]

"""
Feature: Metadata-based ID retrieval
Description: Returns vector IDs matching a metadata filter using Pinecone's query() method.
Parameters:
   - index: Pinecone index object
   - filter_condition: dict, e.g., {"migrated": {"$ne": True}}
   - namespace: optional namespace
   - top_k: max number of results to return
Returns: List of matching vector IDs
"""
def get_ids_with_metadata_filter(index, filter_condition, namespace=None, top_k=10000, dimensions=8):
    keys_with_false = []

    for key in id_db.keys():
        value = id_db[key]
        if id_db[key] == b"false":
            keys_with_false.append(key)
            if len(keys_with_false) >= top_k:
                break

    return keys_with_false

"""
Feature: Reset migration metadata
Description: Removes 'migrated' and 'migrated_at' flags from all vectors in the source index.
Parameters:
    - index: Pinecone Index object
    - namespace: Namespace string
"""
def reset_migrated_flags(index, namespace=None, batch_size=100):
    try:
        #os.remove("./index_list.db")
        keys = id_db.keys()

        for k in keys:
            id_db[k] = b"false"
    except Exception as e:
        logging.error(f"Pagination error while listing vector IDs: {e}")

    print("Reset complete, enter q and Enter to exit")

"""
Feature: Retrieve all vector IDs with pagination
Description: Safely fetches all vector IDs from a Pinecone index using pagination.
Parameters:
    - index: Pinecone Index object
    - namespace: Optional namespace string
Returns: List of vector IDs
"""
def get_source_vector_ids(index, namespace=None):
    non_migrated = []
    vectors = id_db.keys()
    for vector_id in vectors:
        if id_db[vector_id] == b"false":
            non_migrated.append(vector_id)

    return non_migrated

"""
Feature: Filter non-migrated vectors
Description: Checks each vector's metadata to determine if it has not been marked as migrated.
Parameters:
    - index: Pinecone Index object
    - ids: List of vector IDs
    - namespace: Namespace string
Returns: List of vector IDs that have not been migrated
"""
def get_non_migrated_vector_ids(index, ids, namespace=None):
    non_migrated = []
    vectors = id_db.keys()
    for vector_id in vectors:
        if id_db[vector_id] == b"false":
            non_migrated.append(vector_id)

    return non_migrated

#stream instead
def iter_non_migrated_ids():
    for vid in id_db.keys():
        if id_db[vid] == b"false":
            yield vid

"""
Feature: Fetch vectors from index
Description: Retrieves vectors from Pinecone in batches.
Parameters:
    - index: Pinecone Index object
    - ids: List of vector IDs
    - namespace: Namespace string
    - batch_size: Number of vectors to fetch per request
Returns: Dictionary of vector ID -> Vector object
"""
def fetch_vectors(index, ids, namespace=None, batch_size=100):
    from itertools import islice

    def chunked(iterable, size):
        it = iter(iterable)
        return iter(lambda: list(islice(it, size)), [])

    results = {}
    for chunk in chunked(ids, batch_size):
        try:
            res = index.fetch(ids=chunk, namespace=namespace)
            results.update(res.vectors)
        except Exception as e:
            logging.error(f"Fetch error on batch of size {len(chunk)}: {e}")
    return results

"""
Feature: Upsert and tag vectors
Description: Applies migration flags and upserts both to target and source indexes.
Parameters:
    - target_index: Pinecone target index object
    - vectors: Dictionary of vector ID -> Vector object
"""
def upsert_vectors_batch_namespace(target_index, vectors, source_namespace, target_namespace):
    current_batch = []
    current_size = 0
    migrated_timestamp = datetime.utcnow().isoformat()

    def batch_size(vec):
        return len(json.dumps({"id": vec[0], "values": vec[1], "metadata": vec[2]}).encode("utf-8"))

    def flush(batch, source_namespace, target_namespace):
        nonlocal current_batch, current_size
        if batch:
            try:
                target_index.upsert(vectors=batch, namespace=target_namespace)

                logging.info(f"Upserted {len(batch)} vectors")
            except Exception as e:
                logging.error(f"Upsert error: {e}")
            current_batch = []
            current_size = 0

    for vid, v in vectors.items():
        if not v.metadata:
            v.metadata = {}

        item = (vid, v.values, v.metadata)
        size = batch_size(item)
        if current_size + size > MAX_PAYLOAD_BYTES:
            flush(current_batch, source_namespace, target_namespace)
        current_batch.append(item)
        current_size += size

    flush(current_batch, source_namespace, target_namespace)

"""
Feature: Ensure index availability
Description: Verifies or creates the target index with appropriate config.
Parameters:
    - pc: Pinecone client
    - source_index: Pinecone source index
    - target_index_name: Name of target index
"""
def ensure_target_index(pc, target_index_name, dimensions):
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if target_index_name in existing_indexes:
        logging.info(f"Target index '{target_index_name}' already exists.")
        return

    logging.warning(f"Target index '{target_index_name}' does not exist.")

    try:
        pc.create_index(
            name=target_index_name,
            dimension=dimensions,
            metric="cosine",
            spec={"serverless": {"cloud": CLOUD, "region": REGION}}
        )
        logging.info(f"Created target index '{target_index_name}' in {REGION}.")
    except Exception as e:
        logging.error(f"Failed to create index: {e}")
        exit(1)

def process_batch(params):
    batch_ids, source_namespace, target_namespace = params
    try:
        batch_ids, source_namespace, target_namespace = params
        src_index = _worker["src_index"]
        tgt_index = _worker["tgt_index"]

        vectors = fetch_vectors(src_index, batch_ids, namespace=source_namespace, batch_size=BATCH_SIZE)

        if WRITE_TO_PARQUET:
            write_vectors_to_parquet(
                vectors=vectors,
                source_index=src_index,
                namespace=source_namespace,
                output_dir=target_namespace
            )
        else:
            upsert_vectors_batch_namespace(
                target_index=tgt_index,
                vectors=vectors,
                source_namespace=source_namespace,
                target_namespace=target_namespace
            )

        return list(vectors.keys())
    except Exception as e:
        print(f"CRITICAL ERROR in worker: {e}")
        traceback.print_exc()
        return []

def batch_chunks(iterable, size):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])

def progress_listener(queue):
    with tqdm(unit="records", unit_scale=True) as pbar:
        while True:
            try:
                msg = queue.get()
            except EOFError:
                break
            if msg is None:
                break
            pbar.update(msg)

"""
Feature: Execute full migration loop
Description: Main driver function for delta/full migration, batching, and threading.
Parameters:
    - source_index: Pinecone source index object
"""
def migration_loop(source_index, batch_size_loop, dim, source_namespace, target_namespace):
    logging.info("Starting Pinecone Index Migration")

    #all_ids = get_source_vector_ids(source_index, namespace=source_namespace)
    #logging.info("Preparing full migration (dbm-backed id list)")

    batch_size_loop = 2000  # or 500
    start = time.time()

    #handle multiprocessing
    queue = multiprocessing.Queue()

    #keep track with tdm
    listener = multiprocessing.Process(target=progress_listener, args=(queue,))
    listener.start()

    completed_total = 0

    #break into batches
    ids_iter = iter_non_migrated_ids()
    batches = batch_chunks(iterable=ids_iter, size=batch_size_loop)

    #batches = batch_chunks(iterable=all_ids, size=batch_size_loop)
    params = ((batch, source_namespace, target_namespace) for batch in batches)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
        for completed_ids in executor.map(process_batch, params, chunksize=1):
            for vid in completed_ids:
                id_db[vid] = b"true"
            completed_total += len(completed_ids)
            queue.put(len(completed_ids))

    queue.put(None)
    listener.join()

    # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     futures = [executor.submit(process_batch, batch, target_index, source_namespace, target_namespace) for batch in batches]
    #     for f in as_completed(futures):
    #         f.result()
    duration = time.time() - start
    logging.info(f"Migrated {completed_total} vectors in {duration:.2f}s")

#############################################################
#  Main
#############################################################
if __name__ == "__main__":
    source_namespace = SOURCE_NAMESPACE
    target_namespace = TARGET_NAMESPACE

    id_db = dbm.open('index_list', 'c')

    if RESET_MIGRATION_FLAGS:
        reset_migrated_flags(index=source_index, namespace=source_namespace)
    else:
        runloop = True

        # Use describe_index_stats to short-circuit if index is empty
        stats = source_index.describe_index_stats()
        namespaces_keys = list(source_index.describe_index_stats()["namespaces"].keys())
        #start at the first namespace if namespae iteration is turned on
        if ITERATE_OVER_NAMESPACES:
            source_namespace = namespaces_keys[0]

        namespace_stats = stats.namespaces.get(source_namespace, {})

        vector_count = namespace_stats.get("vector_count", 0)
        dimensions = stats.dimension

        #create a new index if the target index does not exist.
        if CREATE_TARGET:
            #updated to use target client
            ensure_target_index(pc=pc_target, target_index_name=TARGET_INDEX_NAME, dimensions=dimensions)

        #this will throw an error if autocreate is not enabled. I'm OK with this.
        target_index = pc_target.Index(TARGET_INDEX_NAME)

        if vector_count == 0:
            logging.info(f"No vectors found in source namespace '{source_namespace}'. Nothing to migrate.")
        else:
            while runloop:
                if ITERATE_OVER_NAMESPACES:
                    if next_namespace_flag:
                        i = namespaces_keys.index(source_namespace)
                        l = len(namespaces_keys)
                        if i >= (l - 1):
                            source_namespace = namespaces_keys[0]
                        else:
                            source_namespace = namespaces_keys[namespaces_keys.index(source_namespace) + 1]
                        next_namespace_flag = False
                        logging.info(f"Next Namespace: {source_namespace}")

                if USE_SOURCE_AS_TARGET_NAMESPACE:
                    target_namespace = source_namespace

                migration_loop(source_index=source_index, batch_size_loop=BATCH_SIZE, dim=dimensions, source_namespace=source_namespace, target_namespace=target_namespace)
                runloop = False

        if hasattr(id_db, "sync"):
            id_db.sync()
        id_db.close()