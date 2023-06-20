import pandas as pd
import re

import functools
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class DictDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.size = min([len(v) for v in data.values()])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class PaddingCollate:
    def __init__(self, padding_keys, pad_token_id=0):
        self.padding_keys = padding_keys
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        outputs = dict_union(batch)
        for k in self.padding_keys:
            outputs[k] = pad_sequence(
                outputs[k], batch_first=True, padding_value=self.pad_token_id
            )
        return outputs


def dict_union(list_of_dicts):
    outputs = {}
    for data in list_of_dicts:
        for k, v in data.items():
            if k in outputs:
                outputs[k].append(v)
            else:
                outputs[k] = [v]
    return outputs


def remove_vowel(text):
    p = re.compile("[eoaiu]")
    return p.sub("", text).strip()


def load_dataset(path, tokenizer):
    data = pd.read_csv(path)
    src_data = tokenizer.batch_encode(data.text.apply(remove_vowel))
    tgt_data = tokenizer.batch_encode(data.text)
    dataset = DictDataset({"src_ids": src_data, "tgt_ids": tgt_data})
    return dataset


def load_dataloader(train_path, valid_path, tokenizer, batch_size):
    train_dataset = load_dataset(train_path, tokenizer)
    valid_dataset = load_dataset(valid_path, tokenizer)
    collate_fn = PaddingCollate(
        ["src_ids", "tgt_ids"], pad_token_id=tokenizer.pad_token_id
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    return train_dataloader, valid_dataloader
