import os
import glob
import sys
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

IN_DIR = "/content/gearnet_chunks"
OUT_PATH = "/content/gearnet_all.pt"

def log(msg):
    print(msg, flush=True)

def main():
    shard_dirs = sorted(d for d in glob.glob(os.path.join(IN_DIR, "*")) if os.path.isdir(d))
    all_data = []

    for shard_dir in shard_dirs:
        success = os.path.join(shard_dir, "_SUCCESS")
        if not os.path.exists(success):
            log(f"[skip] no _SUCCESS in {shard_dir}")
            continue

        chunk_files = sorted(glob.glob(os.path.join(shard_dir, "chunk_*.pt")))
        log(f"[merge] {os.path.basename(shard_dir)} chunks={len(chunk_files)}")

        for chunk_file in chunk_files:
            all_data.extend(torch.load(chunk_file))

    torch.save(all_data, OUT_PATH)
    log(f"[done] wrote {OUT_PATH} n={len(all_data)}")

if __name__ == "__main__":
    main()