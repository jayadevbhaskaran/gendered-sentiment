from config import Config
import utils

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
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
#Note: Last 40 BERT predictions are for control experiment, i.e.,
#"This man is a person."/"This woman is a person."
#First 800 are as earlier, from corpus.
df = pd.read_csv(Config.BERT_FILE, sep="\t", encoding="utf8", header=None, names=["neg", "pos"])
preds2_with_control = list(np.array(df["pos"]).astype("float32"))
preds2 = preds2_with_control[:800]

(t2, prob2, diff2) = utils.ttest(preds2)
results.append([t2, prob2, diff2])

print(results)
np.savetxt(Config.RESULTS_FILE, np.array(results), 
           header="t prob f-m (models: logreg lstm bert)")
print("\n")
###############################################################################
#Gender differences in SST-2 train dataset
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
print("Male: ", len(m_list), np.mean(m_list))       

f_list = []
for item in d:
    words = item["text"].split()
    for old_f in Config.FEMALE_NOUNS:
        f = old_f.replace("This ", "").replace("My ", "").replace("The ", "").replace(", you are ", "").lower()
        if f in words:
           f_list.append(item['_1'])
           break
print("Female: ", len(f_list), np.mean(f_list))   
print("\n")

###############################################################################
#Ad-hoc analysis for each profession
#1. Overall mean positive probability (perception of profession)
#2. Diff. b/w female and male per profession (occupational gender stereotypes)
print("Profession mean_sentiment female-male")
i = 0
map0 = {}
map1 = {}
map2 = {}
for p in Config.PROFESSIONS:
    male0 = preds0[20*i:20*(i+1)]
    male1 = preds1[20*i:20*(i+1)]
    male2 = preds2[20*i:20*(i+1)]

    female0 = preds0[400 + 20*i:400 + 20*(i+1)]
    female1 = preds1[400 + 20*i:400 + 20*(i+1)]
    female2 = preds2[400 + 20*i:400 + 20*(i+1)]
    
    #print(p, np.mean(female0 + male0), np.mean(female0) - np.mean(male0))
    #print(p, np.mean(female1 + male1), np.mean(female1) - np.mean(male1))
    print(p, np.mean(female2 + male2), np.mean(female2) - np.mean(male2))
    
    i = i+1

control_preds = []
control = preds2_with_control[-40:]
male_c = control[:20]
female_c = control[20:]
print("CONTROL", np.mean(female_c + male_c), np.mean(female_c) - np.mean(male_c))
print("\n")
###############################################################################
#Comparing between male/female (e.g., bachelor/spinster)
sentences = utils.get_sentences()
print("noun female-male")
for noun in Config.MALE_NOUNS:
    noun_preds = []
    i = 0
    for sentence in sentences:
        if noun in sentence:
            noun_preds.append(preds2[400+i] - preds2[i])
        i = i+1
    print(noun, np.mean(noun_preds))
   
#Analyze bachelor vs. spinster
bachelor = []
spinster = []
i = 0
for sentence in sentences:
    if "bachelor" in sentence:
        bachelor.append(preds2[i])
        spinster.append(preds2[400+i])
    i = i +1
bs = bachelor + spinster
(t, prob, diff) = utils.ttest(bs)
print("spinster-bachelor: ", diff, prob)
###############################################################################
#Generate plot
df = pd.read_csv(Config.PLOT_DATA_FILE, encoding="utf8")
df = df.dropna(thresh=3)
profession = list(np.array(df["Profession"]).astype("str"))
sentiment = list(np.array(df["Sentiment"]).astype("float32"))
earnings = list(np.array(df["Median Weekly Earnings"]).astype("int"))

fig, ax = plt.subplots()
ax.scatter(sentiment, earnings)

for i, txt in enumerate(profession):
    ax.annotate(txt, (sentiment[i] + 0.01, earnings[i] - 50))

plt.xlabel("Mean predicted positive class probability (BERT)")
plt.ylabel("Median Weekly Earnings (USD)")

z = np.polyfit(sentiment, earnings, 1)
p = np.poly1d(z)
plt.plot(sentiment, p(sentiment), linewidth=0.5)
plt.tight_layout()
plt.savefig(str(Config.PLOT_FILE))