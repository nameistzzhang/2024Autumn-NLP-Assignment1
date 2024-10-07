from vocabdict import VocabDict
from train import Train

DEV_PATH = "./data/dev.txt"
TRAIN_PATH = "./data/train.txt"
TEST_PATH = "./data/test.txt"
VOCABDICT_PATH = "./model/vocab/vocabset.csv"

config = {
    "project": "nlp_homework1",
    "run_name": "run_202410071440",
    "batch_size": 64,
    "timeseq_len": 32,
    "lr": 0.001,
    "embedding_dim": 128,
    "class_num": 4,
    "filter_windows": [3, 4, 5],
    "filter_num": 100,
    "dropout_p": 0.5,
    "l2_constraint": 3.0,
    "linear_hidden_layers": [512, 64],
    "device": "auto", 
    "traindata_path": TRAIN_PATH,
    "testdata_path": TEST_PATH,
    "valdata_path": DEV_PATH,
    "patience": 8,
    "epochs": 100
}

if __name__ == "__main__":

    vocab_dict = VocabDict.loadVocabDict(VOCABDICT_PATH)
    
    train = Train(config, vocab_dict)

    train.train()