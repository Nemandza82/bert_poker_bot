import math
import time
import pandas as pd

nrows = 1024
skip_rows = 10 * 1024 * 1024
#df = pd.read_csv("data/acpc_train.txt", sep=";", names=["score", "villian_cards", "text"], skiprows=skip_rows, nrows=nrows)
df = pd.read_csv("data/acpc_train.txt", sep=";", names=["score", "villian_cards", "text"], skiprows=10000000)


start = time.time()

print(df.iloc[1000000:1000020])

end = time.time()



print(f"Elapsed: {(end - start):.2f}s")
