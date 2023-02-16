import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor


class SenGenDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        input_max_token_len: int,
        output_max_token_len: int,
    ):

        self.tokenizer = tokenizer
        self.data = data
        self.input_max_token_len = input_max_token_len
        self.output_max_token_len = output_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        data_in = data_row["data_in"]

        data_encoding = self.tokenizer(
            data_in,
            max_length=self.input_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_encoding = self.tokenizer(
            data_row["text"],
            max_length=self.output_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = text_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            data_in=data_in,
            text=data_row["text"],
            data_input_ids=data_encoding["input_ids"].flatten(),
            data_attention_mask=data_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=text_encoding["attention_mask"].flatten(),
        )


class SenGenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer,
        batch_size: int,
        input_max_token_len: int,
        output_max_token_len: int,
    ):

        super().__init__()

        self.train_df = train_df
        self.dev_df = dev_df
        self.test_df = test_df

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.input_max_token_len = input_max_token_len
        self.output_max_token_len = output_max_token_len

    def setup(self, stage=None):
        self.train_dataset = SenGenDataset(
            self.train_df,
            self.tokenizer,
            self.input_max_token_len,
            self.output_max_token_len,
        )
        self.dev_dataset = SenGenDataset(
            self.dev_df,
            self.tokenizer,
            self.input_max_token_len,
            self.output_max_token_len,
        )
        self.test_dataset = SenGenDataset(
            self.test_df,
            self.tokenizer,
            self.input_max_token_len,
            self.output_max_token_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=56
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset, batch_size=self.batch_size, shuffle=False, num_workers=56
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=56
        )


class SenGenModel(pl.LightningModule):
    def __init__(
        self, model, tokenizer, input_max_len, output_max_len, optimizer, learning_rate
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.tokenizer = tokenizer
        self.input_max_len = input_max_len
        self.output_max_len = output_max_len
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):

        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def step(self, batch, batch_idx):
        input_ids = batch["data_input_ids"]
        attention_mask = batch["data_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        return self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        loss, outputs = self.step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs = self.step(batch, batch_idx)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == "Adafactor":
            return Adafactor(
                self.parameters(),
                lr=self.learning_rate,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        else:
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def generate_sentence(self, data, gpu=0, min_length=None, do_sample=True):
        data_encoding = self.tokenizer(
            data,
            max_length=self.input_max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            data_encoding.to(torch.device(f"cuda:{gpu}"))

        generated_ids = self.model.generate(
            input_ids=data_encoding["input_ids"],
            attention_mask=data_encoding["attention_mask"],
            max_length=self.output_max_len,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=do_sample,
            min_length=min_length
        )

        preds = [
            self.tokenizer.decode(
                gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for gen_id in generated_ids
        ]

        return "".join(preds)


def get_max_lengths(train_df, dev_df, test_df, tokenizer):
    full_data = pd.concat([train_df, dev_df, test_df])
    data_lens = list()
    text_lens = list()
    for idx, row in full_data.iterrows():
        data_encoding = tokenizer(row["data_in"])
        data_lens.append(len(data_encoding.tokens()))
        text_encoding = tokenizer(row["text"])
        text_lens.append(len(text_encoding.tokens()))
    return (max(data_lens), max(text_lens))

def serialize(dataset: dict) -> str:
    # sort by keys
    data = sorted([(str(k), str(v)) for k, v in dataset.items()])
    # serialize list of tuples to str
    return "; ".join(["➤".join([k, v]) for k, v in data])