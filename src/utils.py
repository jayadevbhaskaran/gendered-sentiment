from config import Config

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind

def get_sentences(gender=None):
    df = pd.read_csv(Config.TSV_TEST_GENDER, sep="\t", encoding="utf8")
    sentences = np.array(df["sentence"]).astype("str")
    g_count = int(len(sentences) / 2)
    if gender == Config.GENDER_M:
        return sentences[:g_count]
    elif gender == Config.GENDER_F:
        return sentences[g_count:]
    return sentences

def ttest(preds):
    n = int(len(preds) / 2)
    male_probs = preds[:n]
    female_probs = preds[n:]
    (t, prob) = ttest_ind(male_probs, female_probs)
    print(np.mean(male_probs), np.mean(female_probs), prob)
    return(t, prob)
    
def glove2dict(src_filename):
    data = {}
    with open(src_filename, encoding="utf8") as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data        