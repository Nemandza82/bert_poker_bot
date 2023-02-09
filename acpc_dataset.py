import torch
import math
import time
import random
import numpy as np
import pandas as pd
from loguru import logger


"""
sep=";"
names=["score", "villian_cards", "street", text"]
"""
def parse_line(line):
    splitted = line.split(";")
    res = {}

    res["score"] = float(splitted[0])
    res["villian_cards"] = splitted[1]
    res["street"] = int(splitted[2])
    res["text"] = splitted[3].strip()

    return res


def is_line_in_street(line, street):
    if street < 0:
        return True
    else:
        return line["street"] == street


"""
Loads random nrows from dataset. From given street (0, 1, 2, 3). If street is -1 any street is taken.
"""
def load_random_df(path, nrows, street=-1, skip_folds=True):
    logger.info(f"Loading random {nrows} from {path}")
    logger.info(f"Counting number of lines in {path}")
    
    line_count = 0
    start = time.time()

    # With buffering=1, should buffer only one line ahead
    with open(path, 'r', buffering=1) as fp:
        
        # Skip first line its header
        fp.readline()

        for line in fp:
            parsed_line = parse_line(line)

            if skip_folds:
                if 'folds' in line:
                    continue

            # Count only lines in required street
            if is_line_in_street(parsed_line, street):
                line_count += 1

    duration = time.time() - start
    logger.info(f"{line_count} number of lines counted in street {street} for {duration:.2f}s")

    line_prob = nrows / line_count
    logger.info(f"Take each line with prob {line_prob:.2f}")

    start = time.time()
    df = []

    while len(df) < nrows:

        # With buffering=1, should buffer only one line ahead
        with open(path, 'r', buffering=1) as fp:

            # Skip first line its header
            fp.readline()

            for line in fp:
                parsed_line = parse_line(line)

                if skip_folds and 'folds' in line:
                    continue

                if not is_line_in_street(parsed_line, street):
                    continue

                if random.random() <= line_prob:
                    df.append(parsed_line)

                if len(df) >= nrows:
                    break

    random.shuffle(df)

    duration = time.time() - start
    logger.info(f"Loaded {nrows} in {duration:.2f}s")

    return df


class AcpcDataset(torch.utils.data.Dataset):
    def __init__(self, df, model):

        self.model = model
        self.labels = [line["score"] for line in df]
        self.texts = [line["text"] for line in df]

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        # print(f"Getting labels: {idx}")
        # print(self.labels[idx])
        x = self.labels[idx]

        # Learn only -1, 1
        #if x != 0:
        #    x = -1 if x < 0 else 1

        #return np.array(math.tanh(x))
        return np.array(x)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        text = self.texts[idx]
        tokenized = self.model.tokenize(text)

        # print(f"Getting texts {idx}")
        # print(f"->{text}<-")
        # print(tokenized)

        return tokenized, text

    def __getitem__(self, idx):

        batch_texts, texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y, texts
