from config import Config

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel

def get_sentences(gender=None):
    df = pd.read_csv(Config.TSV_TEST_GENDER, sep="\t", encoding="utf8")
    sentences = np.array(df["sentence"]).astype("str")
    g_count = int(len(sentences) / 2)
    if gender == Config.GENDER_M:
        return sentences[:g_count]
    elif gender == Config.GENDER_F:
        return sentences[g_count:]
    return sentences

def get_control_sentences():
    sentences = get_sentences()
    m_ = sentences[:20]
    f_ = sentences[400:420]
    m_control = [s.replace("doctor", "person") for s in m_]
    f_control = [s.replace("doctor", "person") for s in f_]
    return m_control + f_control
    
def ttest(preds):
    n = int(len(preds) / 2)
    male_probs = preds[:n]
    female_probs = preds[n:]
    (t, prob) = ttest_rel(male_probs, female_probs)
    diff = np.mean(female_probs) - np.mean(male_probs)
    #print(np.mean(male_probs), np.mean(female_probs), diff, prob)
    return (t, prob, diff)
    
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