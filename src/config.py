from os.path import dirname, realpath, join
from pathlib import Path

class Config:
    BASE_DIR = Path(realpath(join(dirname(realpath(__file__)), "..")))
    DATA_DIR = Path(join(BASE_DIR, "data"))
    RUNS_DIR = Path(join(BASE_DIR, "runs"))
    GLOVE_DIR = Path(join(DATA_DIR, "glove.6b"))
    GLOVE_FILE = Path(join(GLOVE_DIR, "glove.6b.300d.txt"))
    LSTM_MODEL_FILE = Path(join(RUNS_DIR, "lstm.h5"))
    LOGREG_FILE = Path(join(RUNS_DIR, "logreg.txt"))
    LSTM_FILE = Path(join(RUNS_DIR, "lstm.txt"))
    BERT_FILE = Path(join(Path(join(RUNS_DIR, "gender")), "test_results.tsv"))
    RESULTS_FILE = Path(join(RUNS_DIR, "results.txt"))

    TSV_TRAIN = Path(join(DATA_DIR, "train.tsv"))
    TSV_DEV = Path(join(DATA_DIR, "dev.tsv"))
    TSV_TEST_GENDER = Path(join(DATA_DIR, "gender.tsv"))
    PLOT_DATA_FILE = Path(join(DATA_DIR, "plot_data.csv"))
    PLOT_FILE = Path(join(RUNS_DIR, "plot.png"))

    SEED = 42
    MAX_SEQUENCE_LENGTH = 80
    
    GENDER_M = "male"
    GENDER_F = "female"
    GENDERS = [GENDER_F, GENDER_M]
    
    MALE_NOUNS = [
                    "He", "This boy", "This man", "My father", "My son", 
                    "My grandfather", "My grandson", "My uncle", "My nephew",
                    "My brother", "My boyfriend", "My husband", "The groom",
                    "This gentleman", "The male candidate","Sir, you are",
                    "Dad", "Grandpa", "This widower", "This bachelor", 
                ]

    FEMALE_NOUNS = [
                    "She", "This girl", "This woman", "My mother", "My daughter", 
                    "My grandmother", "My granddaughter", "My aunt", "My niece",
                    "My sister", "My girlfriend", "My wife", "The bride",
                    "This lady", "The female candidate","Madam, you are",
                    "Mum", "Grandma", "This widow", "This spinster",                 
                ]
    
    PROFESSIONS = [
            "doctor", "tailor", "baker", "secretary", "professor", "scientist",
            "writer", "teacher", "truck driver", "pilot", "lawyer",
            "flight attendant", "nurse", "chef", "soldier", "dancer",
            "gym trainer", "mechanic", "clerk", "bartender"
            ]
    