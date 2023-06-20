# adapted from pytorch tutorial
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.distributions import Categorical


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout, max_len=120):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        pos_embedding = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        pos_embedding[:, :, 0::2] = torch.sin(X)
        pos_embedding[:, :, 1::2] = torch.cos(X)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, X):
        X = X + self.pos_embedding[:, : X.shape[1], :]
        return self.dropout(X)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        tokenizer,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len=120,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.d_model = d_model
        self.tokenizer = tokenizer
        self.padding_idx = tokenizer.pad_token_id
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            activation=nn.GELU(),
            dropout=dropout,
            norm_first=True,
        )
        self.generator = nn.Linear(d_model, len(tokenizer))
        self.tok_emb = nn.Embedding(
            len(tokenizer), d_model, padding_idx=self.padding_idx
        )
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout, max_len=max_len
        )
        self.generator.weight = self.tok_emb.weight  # weight sharing
        self.init_params()

    def forward(self, src, tgt):
        mem, mk = self.encoder_forward(src)
        logits = self.generator(self.decoder_forward(tgt, mem, mk))
        return logits

    def encoder_forward(self, src):
        x = self.tok_emb(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        src_padding_mask = src == self.padding_idx
        memory = self.transformer.encoder(
            x.transpose(0, 1), src_key_padding_mask=src_padding_mask
        )
        return memory, src_padding_mask

    def decoder_forward(self, tgt, memory, memory_mask):
        x = self.tok_emb(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            x.size(1), self.device
        ).bool()
        out = self.transformer.decoder(
            x.transpose(0, 1),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=(tgt == self.padding_idx),
            memory_key_padding_mask=memory_mask,
        )
        return out.transpose(0, 1)

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @property
    def device(self):
        return next(self.parameters()).device

    def evaluate_actions(self, source_ids, response_ids):
        mem, mk = self.encode(source_ids)
        logits = self.decoder_out(response_ids, mem, mk)
        action_ids = response_ids
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action_ids)
        entropy = dist.entropy()
        value_pred = self.value_fn(logits).squeeze(-1)
        padding_mask = action_ids != self.tokenizer.pad_token_id
        return value_pred, log_probs, entropy, padding_mask

    @torch.no_grad()
    def decode_sentences(self, src_sentences, maxlen=101, greedy=True):
        self.eval()
        src = self.tokenizer(src_sentences).to(self.device)
        is_done = torch.zeros(len(src)).bool().to(self.device)
        memory, src_mask = self.encoder_forward(src)
        ys = (
            torch.full((len(src), 1), self.tokenizer.bos_token_id)
            .long()
            .to(self.device)
        )
        for i in range(maxlen - 1):
            all_logits = self.generator(self.decoder_forward(ys, memory, src_mask))
            last_token_logits = all_logits[:, -1]
            if greedy:
                t_pred = torch.argmax(last_token_logits, dim=1)
            else:
                dist = Categorical(logits=last_token_logits)
                t_pred = dist.sample().flatten()

            cat_this = torch.where(is_done, self.padding_idx, t_pred).unsqueeze(1)
            is_done[t_pred == self.tokenizer.eos_token_id] = True

            ys = torch.cat([ys, cat_this], dim=1)

            if torch.all(is_done):
                break

        return self.tokenizer.batch_decode(ys.cpu().numpy(), remove_special_tokens=True)
