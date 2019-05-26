from config import Config
SEED = Config.SEED

import numpy as np
np.random.seed(SEED)
import pandas as pd
import utils

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

train = pd.read_csv(Config.TSV_TRAIN, sep="\t", header=None, names=["idx", "class", "dummy", "text"])
dev = pd.read_csv(Config.TSV_DEV, sep="\t", header=None, names=["idx", "class", "dummy", "text"])
X_train, X_dev, y_train, y_dev = train["text"], dev["text"], train["class"], dev["class"]
###############################################################################
clf = Pipeline([
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("logreg_bow", LogisticRegression(random_state=SEED, solver="liblinear"))
        ])

clf.fit(X_train, y_train)
dev_preds = clf.predict(X_dev)
print(accuracy_score(y_dev, dev_preds))

sentences = utils.get_sentences()
preds = [item[1] for item in clf.predict_proba(sentences)]
file = Config.LOGREG_FILE
np.savetxt(file, preds, header="dev accuracy: " + str(accuracy_score(y_dev, dev_preds)))
(t, prob, diff) = utils.ttest(preds)
###############################################################################

