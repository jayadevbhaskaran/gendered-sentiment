from config import Config
import utils

import numpy as np
import pandas as pd

df = pd.read_csv(Config.BERT_FILE, sep="\t", encoding="utf8", header=None, names=["neg", "pos"])
preds = list(np.array(df["pos"]).astype("float32"))
(t, prob) = utils.ttest(preds)
