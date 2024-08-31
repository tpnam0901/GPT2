# Dataset link
# https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k


import os
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        default="../working/dataset/openwebtext",
        help="Path to output folder",
    )
    return parser.parse_args()


args = arg_parser()


os.makedirs(args.out_dir, exist_ok=True)

num_proc = 8

num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")


def get_huggingface_dataset_tokens(name, process_fn):
    dataset = load_dataset(name, num_proc=num_proc_load_dataset)
    dataset["train"] = dataset.pop("train_sft")
    dataset["val"] = dataset.pop("test_sft")

    # tokenize the dataset
    tokenized = dataset.map(
        process_fn,
        remove_columns=list(dataset["val"][0].keys()),
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    return tokenized


def process_everyday_conversations(example):
    ids = []
    example = example["messages"]
    for e in example:
        content = e["content"]
        content = content.replace('"', "''")
        content = content.replace("\\", "/")
        ids += enc.encode_ordinary(content)
        ids.append(enc.eot_token)
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {"ids": ids, "len": len(ids)}
    return out


def write_dataset(tokenized, start_idx, start_len, mode="r+"):
    # concatenate all the ids in each dataset into one large file we can use for training
    indices = start_idx
    arr_len = start_len
    for split, dset in tokenized.items():
        arr_len[split] = np.sum(dset["len"], dtype=np.uint64) + arr_len[split]
        filename = os.path.join(os.path.abspath(args.out_dir), f"{split}.bin")
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        print("Allocate memory")
        arr = np.memmap(filename, dtype=dtype, mode=mode, shape=(arr_len[split],))
        total_batches = len(dset["ids"]) // 1024
        print("Start writing")
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[indices[split] : indices[split] + len(arr_batch)] = arr_batch
            indices[split] += len(arr_batch)
        arr.flush()
    return indices, arr_len


chatbot_dataset = {
    "HuggingFaceTB/everyday-conversations-llama3.1-2k": process_everyday_conversations,
}

firstData = True
indices = {"train": 0, "val": 0}
arr_len = {"train": np.uint64(0), "val": np.uint64(0)}
for name, func in chatbot_dataset.items():
    print(f"Process {name}")
    tokenized = get_huggingface_dataset_tokens(name, func)
    mode = "w+" if firstData else "r+"
    print(f"Write {name}")
    indices, arr_len = write_dataset(tokenized, indices, arr_len, mode)
    firstData = False
