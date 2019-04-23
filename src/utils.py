from config import Config

import numpy as np

from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind

def fit_and_evaluate(X_train, X_dev, y_train, y_dev, clf):
    clf.fit(X_train, y_train)
    preds = clf.predict(X_dev)
    print(accuracy_score(y_dev, preds))
    
    all_male_sentences = []
    all_female_sentences = []
    
    for p in Config.PROFESSIONS:
        male_sentences = []
        female_sentences = []
    
        for m in Config.MALE_NOUNS:
            male_sentences.append(m + " is a " + p + ".")
            male_sentences.append(m + " wants to be a " + p + ".")
            all_male_sentences.append(m + " is a " + p + ".")
            all_male_sentences.append(m + " wants to be a " + p + ".")

    
        for f in Config.FEMALE_NOUNS:
            female_sentences.append(f + " is a " + p + ".")
            female_sentences.append(f + " wants to be a " + p + ".")
            all_female_sentences.append(f + " is a " + p + ".")
            all_female_sentences.append(f + " wants to be a " + p + ".")
            
        male_probs = [item[1] for item in clf.predict_proba(male_sentences)]
        female_probs = [item[1] for item in clf.predict_proba(female_sentences)]
        (t, prob) = ttest_ind(male_probs, female_probs)
        print(p, np.mean(male_probs), np.mean(female_probs), prob)
    
    all_male_probs = [item[1] for item in clf.predict_proba(all_male_sentences)]
    all_female_probs = [item[1] for item in clf.predict_proba(all_female_sentences)]
    (t, prob) = ttest_ind(all_male_probs, all_female_probs)
    print("Total", np.mean(all_male_probs), np.mean(all_female_probs), prob)
    print("\n")

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