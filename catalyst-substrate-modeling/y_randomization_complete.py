from pathlib import Path

import re
import numpy as np
import pandas as pd

#Read in data
DATA_FILE = Path("training_set_base.csv")
df = pd.read_csv(DATA_FILE, encoding="utf-8-sig").reset_index(drop=True)

#Define y-shuffle
def y_shuffle(df):
    """
    Randomizes the y-values in the dataframe.
    """
    df_shuffled = df.copy()
    df_shuffled['y'] = np.random.permutation(df_shuffled['y'].values)
    return df_shuffled

#Create N randomized datasets
N = 5
randomized_dfs = [y_shuffle(df) for _ in range(N)]


