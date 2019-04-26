from config import Config
import numpy as np
import pandas as pd

items = []
idx = 0

for p in Config.PROFESSIONS:
    for m in Config.MALE_NOUNS:
        sentence = m + " is a " + p + "."
        if m == "Sir, you are":
            sentence = m + " a " + p + "."
        items.append({"id": idx, "sentence": sentence})
        idx = idx + 1

for p in Config.PROFESSIONS:
    for f in Config.FEMALE_NOUNS:
        sentence = f + " is a " + p + "."
        if f == "Madam, you are":
            sentence = f + " a " + p + "."
        items.append({"id": idx, "sentence": sentence})
        idx = idx + 1

df = pd.DataFrame(items)
df.to_csv(Config.TSV_TEST_GENDER, sep = "\t", index=False, encoding="utf8")