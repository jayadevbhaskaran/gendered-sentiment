from config import Config
import utils

import numpy as np
import pandas as pd

###############################################################################
#Print results
results = []
#LogReg
preds0 = list(np.array(pd.read_csv(Config.LOGREG_FILE, encoding="utf8")).reshape(-1,))
(t0, prob0, diff0) = utils.ttest(preds0)
results.append([t0, prob0, diff0])

#BiLSTM
preds1 = list(np.array(pd.read_csv(Config.LSTM_FILE, encoding="utf8")).reshape(-1,))
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
###############################################################################
#Ad-hoc analysis for gender differences in SST-2 train dataset
train = pd.read_csv(Config.TSV_TRAIN, sep="\t", header=None, names=["idx", "class", "dummy", "text"])
d = train.to_dict("records")
m_list = []
for item in d:
    words = item["text"].split()
    for old_m in Config.MALE_NOUNS:
        m = old_m.replace("This ", "").replace("My ", "").replace("The ", "").replace(", you are ", "").lower()
        if m in words:
           m_list.append(item['_1'])
           break
print(len(m_list), np.mean(m_list))       

f_list = []
for item in d:
    words = item["text"].split()
    for old_f in Config.FEMALE_NOUNS:
        f = old_f.replace("This ", "").replace("My ", "").replace("The ", "").replace(", you are ", "").lower()
        if f in words:
           f_list.append(item['_1'])
           break
print(len(f_list), np.mean(f_list))       
###############################################################################
#Ad-hoc analysis for each profession
i = 0
map0 = {}
map1 = {}
map2 = {}
for p in Config.PROFESSIONS:
    print(p, "\n")
    male0 = preds0[20*i:20*(i+1)]
    male1 = preds1[20*i:20*(i+1)]
    male2 = preds2[20*i:20*(i+1)]

    female0 = preds0[400 + 20*i:400 + 20*(i+1)]
    female1 = preds1[400 + 20*i:400 + 20*(i+1)]
    female2 = preds2[400 + 20*i:400 + 20*(i+1)]
    
    print(np.mean(female0 + male0), np.mean(female0) - np.mean(male0), "\n")
    print(np.mean(female1 + male1), np.mean(female1) - np.mean(male1), "\n")
    print(np.mean(female2 + male2), np.mean(female2) - np.mean(male2), "\n")
    
    i = i+1
###############################################################################


