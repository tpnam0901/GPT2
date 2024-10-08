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
import json


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
        test_size=0.0005, seed=2357, shuffle=True
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


def process_LaMini(example):
    ids = enc.encode_ordinary(
        example["instruction"]
    )  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    ids += enc.encode_ordinary(example["response"])
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    out = {"ids": ids, "len": len(ids)}
    return out


def process_chatbot_arena(example):
    ids = []
    conversation = example["conversation_a"] + example["conversation_b"]
    for data in conversation:
        ids += enc.encode_ordinary(data["content"])
        ids.append(enc.eot_token)
    out = {"ids": ids, "len": len(ids)}
    return out


def process_instruct(example):
    ids = []
    keys = ["prompt", "completion"]
    for k in keys:
        ids += enc.encode_ordinary(example[k])
        ids.append(enc.eot_token)
    out = {"ids": ids, "len": len(ids)}
    return out


def process_finance_alpaca(example):
    ids = []
    keys = [["instruction", "input"], "output"]
    for k in keys:
        if type(k) == list:
            text = example[k[0]] + " " + example[k[1]]
        else:
            text = example[k]
        ids += enc.encode_ordinary(text)
        ids.append(enc.eot_token)
    out = {"ids": ids, "len": len(ids)}
    return out


def process_im_curious(example):
    ids = []
    keys = ["question", "answer"]
    for k in keys:
        ids += enc.encode_ordinary(example[k])
        ids.append(enc.eot_token)
    out = {"ids": ids, "len": len(ids)}
    return out


def process_chatalpaca(example):
    ids = []
    example = example["conversations"]
    for e in example:
        ids += enc.encode_ordinary(e["value"])
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
    "MBZUAI/LaMini-instruction": process_LaMini,
    "lmsys/chatbot_arena_conversations": process_chatbot_arena,
    "swype/instruct": process_instruct,
    "gbharti/finance-alpaca": process_finance_alpaca,
    "xiyuez/im-feeling-curious": process_im_curious,
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

json_path = "chatalpaca-20k.json"
data_json = []
with open(json_path, "r") as user_file:
    lines = user_file.readlines()
for line in lines:
    json_data = json.loads(line)
    data_json.append({"conversations": json_data["conversations"]})
with open("temp.json", "w") as f:
    json.dump(data_json, f)

ds = load_dataset("json", data_files="temp.json", split="train")
os.remove("temp.json")

tokenized = ds.map(
    process_chatalpaca,
    remove_columns=["conversations"],
    desc="tokenizing the splits",
    num_proc=8,
)

write_dataset({"train": tokenized}, indices, arr_len, "r+")
