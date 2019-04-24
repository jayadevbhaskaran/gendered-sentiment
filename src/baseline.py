from config import Config
SEED = Config.SEED

import numpy as np
np.random.seed(SEED)
import pandas as pd
import utils

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
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

utils.fit_and_evaluate(X_train, X_dev, y_train, y_dev, clf)
###############################################################################