# gendered-sentiment

Repository structure:
* data: Contains training datasets (Stanford Sentiment Treebank 2) and our new corpus (plaintext + annotated versions), GloVe (`glove.6b.300d`) vectors need to be downloaded to run the code (stored within `data/glove.6b`)
* runs: Folder to output model predictions (for baseline and biLSTM models), also contains a subfolder with BERT model predictions
* src: Contains scripts needed for data preprocessing, model training, and all ad-hoc analysis used in the paper
