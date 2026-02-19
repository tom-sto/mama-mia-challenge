from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib

def file_checksum(path, algo="sha256", chunk_size=8192):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def folder_checksum_parallel(folder, algo="sha256", workers=None):
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    total = len(files)

    results = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(file_checksum, path, algo): path
            for path in files
        }

        for future in as_completed(future_map):
            path = future_map[future]
            results[path] = future.result()

            completed += 1
            print(f"Hashed {completed}/{total}: {path}", flush=True)

    # Final deterministic combine
    h = hashlib.new(algo)
    for path in files:
        h.update(str(path.relative_to(folder)).encode())
        h.update(b"\0")
        h.update(results[path].encode())

    return h.hexdigest()

print(folder_checksum_parallel(r"/mnt/storageSSD/MAMA-MIA/data/my_preprocessed_data/Dataset106_cropped_Xch_breast_no_norm/training"))
