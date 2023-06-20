import torch


class Trainer:
    def __init__(self, model, loss_fn, optimizer, tokenizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.tokenizer = tokenizer

    def fit(self, epochs, train_dataloader, valid_dataloader):
        sample = next(iter(train_dataloader))
        x_sample = tokenizer.batch_decode(sample["src_ids"][:5].numpy())
        y_sample = tokenizer.batch_decode(sample["tgt_ids"][:5].numpy())
        for e in range(epochs):
            train_loss = self.train(train_dataloader)
            valid_loss = self.evaluate(valid_dataloader)

            print(
                "EPOCH: %s :: TRAIN LOSS: %s :: VALID LOSS: %s"
                % (e, train_loss, valid_loss)
            )

            for x, y in zip(model.decode_sentences(x_sample), y_sample):
                print("target:", y)
                print("model :", x)

    def train(self, train_dataloader):
        self.model.train()
        losses = 0
        for batch in train_dataloader:
            src = batch["src_ids"].to(self.model.device)
            tgt = batch["tgt_ids"].to(self.model.device)

            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            y_pred = self.model(src, tgt_input)
            loss = self.loss_fn(
                y_pred.reshape(-1, y_pred.shape[-1]), tgt_out.reshape(-1)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses += loss.item()

        return losses / len(list(train_dataloader))

    @torch.no_grad()
    def evaluate(self, valid_dataloader):
        self.model.eval()
        losses = 0
        for batch in valid_dataloader:
            src = batch["src_ids"].to(self.model.device)
            tgt = batch["tgt_ids"].to(self.model.device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            y_pred = self.model(src, tgt_input)
            loss = self.loss_fn(
                y_pred.reshape(-1, y_pred.shape[-1]), tgt_out.reshape(-1)
            )
            losses += loss.item()

        return losses / len(list(valid_dataloader))


def get_arguments(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/toxic.train.csv")
    parser.add_argument("--valid_path", default="./data/toxic.test.csv")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument("--d_model", default=512, type=int)
    parser.add_argument("--nhead", default=8, type=int)
    parser.add_argument("--dim_feedforward", default=512, type=int)
    parser.add_argument("--num_encoder_layers", default=3, type=int)
    parser.add_argument("--num_decoder_layers", default=3, type=int)

    return parser.parse_args(args=args)


if __name__ == "__main__":
    from src.tokenizer import CharTokenizer
    from src.dataset import load_dataloader
    from src.model import Seq2SeqTransformer

    args = get_arguments()
    tokenizer = CharTokenizer()
    train_dataloader, valid_dataloader = load_dataloader(
        args.train_path, args.valid_path, tokenizer, args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer(
        tokenizer,
        args.num_encoder_layers,
        args.num_decoder_layers,
        args.d_model,
        args.nhead,
        args.dim_feedforward,
    )
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    trainer = Trainer(model, loss_fn, optimizer, tokenizer)
    trainer.fit(
        epochs=args.num_epochs,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
    )
