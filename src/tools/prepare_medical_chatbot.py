# Dataset link
# https://huggingface.co/datasets/MBZUAI/LaMini-instruction?row=1
# https://huggingface.co/datasets/lmsys/chatbot_arena_conversations?row=7
# https://huggingface.co/datasets/swype/instruct
# https://huggingface.co/datasets/gbharti/finance-alpaca
# https://github.com/cascip/ChatAlpaca
# https://huggingface.co/datasets/xiyuez/im-feeling-curious


import os
from tqdm import tqdm
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

    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=1996, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")  # rename

    # tokenize the dataset
    tokenized = split_dataset.map(
        process_fn,
        remove_columns=list(split_dataset["val"][0].keys()),
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    return tokenized


def process(example):
    ids = []
    keys = ["Description", "Patient"]
    for k in keys:
        question = example[k]
        question = question.replace('"', "''")
        question = question.replace("\\", "/")
        ids += enc.encode_ordinary(question)
        ids.append(enc.eot_token)

        answer = example["Doctor"]
        answer = answer.replace('"', "''")
        answer = answer.replace("\\", "/")
        ids += enc.encode_ordinary(answer)
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
        total_batches = len(dset["ids"]) // 128
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
    "ruslanmv/ai-medical-chatbot": process,
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
