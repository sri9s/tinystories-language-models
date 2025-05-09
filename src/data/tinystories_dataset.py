"""Download, preprocess and serve the TinyStories dataset as a DataLoader."""

import argparse
import glob
import json
import math
import os
import random
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from src.data.tokenizer import Tokenizer
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)
DATA_CACHE_DIR = "data"


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url.

    Parameters
    ----------
    url
        URL to download from.
    fname
        Filename to save to.
    chunk_size, optional
        Chunk size in bytes (default: 1024).
    """
    resp = requests.get(url, stream=True, timeout=5)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Download the TinyStories dataset."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        with tarfile.open(data_filename, "r:gz") as tar:
            for member in tar.getmembers():
                # Ensure that the member is a file (not a directory or symlink)
                print(f"Member: {member}")
                if member.isfile():
                    # Check if the filename contains safe characters (modify this as needed)
                    # if not any(char in member.name for char in ["/", "\\", ".."]):
                    tar.extract(member, data_dir)
            # tar.extractall(data_dir)
        # os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")
    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print(f"Shard filenames: {shard_filenames}")
    with open(shard_filenames[0]) as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")


def pretokenize():
    """Pretokenize the TinyStories dataset."""
    enc = Tokenizer()

    def process_shard(shard):
        """Tokenize a single shard and save to disk.

        Parameters
        ----------
        shard
            Path to a single shard (json file).
        """
        with open(shard) as f:
            data = json.load(f)
        all_tokens = []
        for example in tqdm(data):
            text = example["story"]
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        # write to disk
        tokenized_filename = shard.replace(".json", ".bin")
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        print(f"Saved {tokenized_filename}")

    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print(f"Shard filenames: {shard_filenames}")
    # process all the shards in a threadpool
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_shard, shard_filenames)

    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        # print(f"Created a PretokDataset with rng seed {seed}")
        data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        print(f"Shard filenames: {shard_filenames} in {self.split} split")
        while True:
            rng.shuffle(shard_filenames)
            for shard in tqdm(shard_filenames, desc="Shards"):
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "train_tokenizer", "pretokenize"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    fun = {
        "download": download,
        "pretokenize": pretokenize,
    }
    fun[args.stage]()
