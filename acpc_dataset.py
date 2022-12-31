import torch
import math
import numpy as np
import pandas as pd


class AcpcDataset(torch.utils.data.Dataset):
    def __init__(self, path, skip_rows, nrows, tokenizer):

        print(f"Loading {path} dataset")
        df = pd.read_csv(path, sep=";", skiprows=skip_rows, nrows=nrows)

        self.tokenizer = tokenizer
        self.labels = [float(label) for label in df['score']]
        self.texts = df['text']

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        #print(f"Getting labels: {idx}")
        #print(self.labels[idx])
        x = self.labels[idx]

        return np.array(math.tanh(x))

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        text = self.texts[idx]
        tokenized = self.tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

        #print(f"Getting texts {idx}")
        #print(f"->{text}<-")
        #print(tokenized)

        return tokenized

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
