#!/usr/bin/env python3

# adapted from: https://github.com/KellerJordan/modded-nanogpt/blob/a202a3a0ca99d69bb7f847e5337c7c6e0890fd92/data/cached_fineweb10B.py

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

DEFAULT_CHUNKS = 8
DEFAULT_DATA_DIR = "fineweb10B"

def download_dataset(dir:str | Path | None, chunks=DEFAULT_CHUNKS):
    assert 0 < chunks <= 103

    hf_hub_download("kjj0/fineweb10B-gpt2", "fineweb_val_000000.bin", repo_type="dataset", local_dir=dir)

    for i in range(1, chunks + 1):
        hf_hub_download("kjj0/fineweb10B-gpt2", f"fineweb_train_{i:06d}.bin", repo_type="dataset", local_dir=dir)

if __name__ == "__main__":
    def main():
        parser = argparse.ArgumentParser(description='Download Fineweb10B tokens')
        parser.add_argument('--chunks', type=int, default=DEFAULT_CHUNKS, help=f'Number of chunks (default: {DEFAULT_CHUNKS}, max: 103)')
        parser.add_argument('--dir', type=str, default=DEFAULT_DATA_DIR, help=f'Local directory (default: {DEFAULT_DATA_DIR})')
        args = parser.parse_args()
        download_dataset(args.chunks, args.dir)
    main()
