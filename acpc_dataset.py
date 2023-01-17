import torch
import math
import time
import random
import numpy as np
import pandas as pd
from loguru import logger


def load_random_df(path, nrows):
    logger.info(f"Loading random {nrows} from {path}")
    logger.info(f"Counting number of lines in {path}")
    
    line_count = 0
    start = time.time()

    with open(path, 'r') as fp:
        for line in fp:
            line_count += 1

    duration = time.time() - start
    logger.info(f"Number of lines {line_count} counted in {duration:.2f}s")

    skip_rows = random.randint(0, line_count - nrows - 10)
    logger.info(f"Randomized number of skip rows {skip_rows}")

    #print(f"Loading {path} dataset")
    start = time.time()
    df = pd.read_csv(path, sep=";", names=["score", "villian_cards", "text"], skiprows=skip_rows, nrows=nrows)
    duration = time.time() - start

    logger.info(f"Loaded dataframe in {duration:.2f}s")

    #print(f"Loaded csv {path}")
    #print(df)

    return df


class AcpcDataset(torch.utils.data.Dataset):
    def __init__(self, df, model):

        self.model = model
        self.labels = [float(label) for label in df["score"]]
        self.texts = [text for text in df["text"]]

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        # print(f"Getting labels: {idx}")
        # print(self.labels[idx])
        x = self.labels[idx]

        return np.array(math.tanh(x))

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        text = self.texts[idx]
        tokenized = self.model.tokenize(text)

        # print(f"Getting texts {idx}")
        # print(f"->{text}<-")
        # print(tokenized)

        return tokenized

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
