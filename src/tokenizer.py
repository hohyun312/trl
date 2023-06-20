import torch
from torch.nn.utils.rnn import pad_sequence
import re


class CharTokenizer:
    def __init__(self):
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.padding_side = "right"
        self.tokens_to_ids = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2}
        self.ids_to_tokens = {0: "[PAD]", 1: "[BOS]", 2: "[EOS]"}
        self.update_vocab(range(10))
        self.update_vocab("abcdefghijklmnopqrstuvwxyz")
        self.update_vocab(",.?!\"' ")

    def convert_ids_to_tokens(self, ids):
        outputs = []
        for i in ids:
            outputs.append(self.ids_to_tokens[i])
        return outputs

    def convert_tokens_to_ids(self, tokens):
        outputs = []
        for i in tokens:
            outputs.append(self.tokens_to_ids[i])
        return outputs

    def encode(self, text):
        tokens = ["[BOS]"] + list(text) + ["[EOS]"]
        ids = self.convert_tokens_to_ids(tokens)
        return torch.LongTensor(ids)

    def batch_encode(self, batched_text):
        batched_ids = [self.encode(text) for text in batched_text]
        return batched_ids

    def decode(self, ids):
        return "".join(self.convert_ids_to_tokens(ids))

    def batch_decode(self, batched_ids, remove_special_tokens=True):
        batched_text = [self.decode(ids) for ids in batched_ids]
        if remove_special_tokens:
            p = re.compile("\[BOS\]|\[EOS\]|\[PAD\]")
            batched_text = [p.sub("", txt) for txt in batched_text]
        return batched_text

    def update_vocab(self, vocab):
        self.tokens_to_ids.update(
            {str(v): i + len(self.tokens_to_ids) for i, v in enumerate(vocab)}
        )
        self.ids_to_tokens.update(
            {i + len(self.ids_to_tokens): str(v) for i, v in enumerate(vocab)}
        )

    def __call__(self, batched_text):
        batched_ids = self.batch_encode(batched_text)
        return pad_sequence(
            batched_ids, batch_first=True, padding_value=self.pad_token_id
        )

    def __len__(self):
        return len(self.tokens_to_ids)
