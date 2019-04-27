from config import Config
import utils

import numpy as np
import pandas as pd

results = []

#LogReg
preds0 = list(np.array(pd.read_csv(Config.LOGREG_FILE, encoding="utf8")).reshape(-1,))
(t0, prob0, diff0) = utils.ttest(preds0)
results.append([t0, prob0, diff0])

#BiLSTM
preds1 = list(np.array(pd.read_csv(Config.LSTM_FILE, sep="\t", encoding="utf8")).reshape(-1))
(t1, prob1, diff1) = utils.ttest(preds1)
results.append([t1, prob1, diff1])

#BERT
df = pd.read_csv(Config.BERT_FILE, sep="\t", encoding="utf8", header=None, names=["neg", "pos"])
preds2 = list(np.array(df["pos"]).astype("float32"))
(t2, prob2, diff2) = utils.ttest(preds2)
results.append([t2, prob2, diff2])

print(results)
np.savetxt(Config.RESULTS_FILE, np.array(results), 
           header="t prob f-m (models: logreg lstm bert)")




