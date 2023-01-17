import math
import time
import pandas as pd

nrows = 1024
skip_rows = 10 * 1024 * 1024
#df = pd.read_csv("data/acpc_train.txt", sep=";", names=["score", "villian_cards", "text"], skiprows=skip_rows, nrows=nrows)
df = pd.read_csv("data/acpc_train.txt", sep=";", names=["score", "villian_cards", "text"], skiprows=10000000)


skip_rows = 10
nrows = 10

start = time.time()

batch_df = df.iloc[skip_rows:skip_rows+nrows]
print(batch_df)

end = time.time()

print(len(batch_df.index))

texts = batch_df["text"]
print(f"Text 0: {texts[0]}")

print(f"Elapsed: {(end - start):.2f}s")
