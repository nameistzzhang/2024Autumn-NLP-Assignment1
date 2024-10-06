import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pandas as pd
import csv

DEV_PATH = "./data/dev.txt"
TRAIN_PATH = "./data/train.txt"
TEST_PATH = "./data/test.txt"
VOCABDICT_PATH = "./model/vocab/vocabset.csv"

def buildVocabDict(vocabdict_path : str, dataset1_path : str, dataset2_path : str) -> None:

    """
    buildVocabDict: Given the path to the vocabulary dictionary, dataset1 and dataset2, load the vocabulary dictionary and save it to the csv file.
    Parameters:
        vocabdict_path: path to the vocabulary dictionary.
        dataset1_path: path to the dataset1.
        dataset2_path: path to the dataset2.
    Returns:
        None
    """ 

    # open the dataset file; format: text \t label
    with open(dataset1_path, 'r', encoding='utf-8') as f:
        dataset1 = f.readlines()
    dataset1 = [line.strip().split('\t') for line in dataset1]
    with open(dataset2_path, 'r', encoding='utf-8') as f:
        dataset2 = f.readlines()
    dataset2 = [line.strip().split('\t') for line in dataset2]
    dataset = dataset1 + dataset2
    
    # use jieba to tokenize the chinese text
    all_tokens = []
    for text, _ in dataset:
        all_tokens += list(jieba.cut(text))
    # count the frequency of each token
    token_counts = Counter(all_tokens)
    
    vocabulary = {token: idx for idx, (token, _) in enumerate(token_counts.items())}
    print("The length of the vocabulary dictionary is : ", len(vocabulary))

    # save the vocabulary to the csv file
    dataframe = pd.DataFrame(vocabulary.items(), columns=['token', 'idx'])
    dataframe.to_csv(vocabdict_path, index=False, sep=',')

    return

def loadVocabDict(vocabdict_path : str) -> dict:

    """
    loadVocabDict: Given the path to the vocabulary dictionary, load the vocabulary dictionary from the csv file.
    Parameters:
        vocabdict_path: path to the vocabulary dictionary.
    Returns:
        dict: the vocabulary dictionary.
    """

    # build an empty dictionary
    vocab_dict = {}

    # read from csv file
    with open(vocabdict_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            token, idx = row
            vocab_dict[token] = int(idx)

    return vocab_dict

class model(nn.Module):

    def __init__(self, 
                 vocab_dict : str, 
                 embedding_dim : str = 128, 
                 class_num : str = 4, 
                 filter_windows : list = [3, 4, 5], 
                 filter_num : int = 100, 
                 dropout_p : float = 0.5, 
                 l2_constraint : float = 3.0, 
                 linear_hidden_layers : list = [512, 64], 
                 device : str = "auto") -> None:
        super(model, self).__init__()

        self.vocab_dict_size = len(vocab_dict) + 1
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.embedding = nn.Embedding(self.vocab_dict_size, embedding_dim).to(self.device)
        self.convs = nn.ModuleList([nn.Conv2d(
            in_channels = 1, 
            out_channels=filter_num, 
            kernel_size=(window, embedding_dim))for window in filter_windows]).to(self.device)
        self.maxpools = nn.ModuleList([nn.AdaptiveMaxPool1d(1) for _ in filter_windows]).to(self.device)
        self.classifier = nn.Sequential(
            nn.Linear(filter_num * len(filter_windows), linear_hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(linear_hidden_layers[0], linear_hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(linear_hidden_layers[1], class_num)
        ).to(self.device)

        self.l2_constraint = l2_constraint

    def sentence2Input(self, sentence : str) -> torch.Tensor:

        # get a random sentence and tokenize it
        sentence = list(jieba.cut(sentence))
        sentence_idx = []
        for token in sentence:
            if token in vocab_dict:
                sentence_idx.append(vocab_dict[token])
            else:    # which means unkonwn token
                sentence_idx.append(self.vocab_dict_size - 1)
        
        # token embedding
        sentence_idx = torch.tensor([sentence_idx])

        return sentence_idx

    def forward(self, x):

        x = self.embedding(x)
        print("after embedding : ", x.shape)
        x = x.unsqueeze(1)
        print("after unsqueeze : ", x.shape)
        x = [F.relu(conv(x)).squeeze(-1) for conv in self.convs]
        print("after conv : ", x[0].shape)
        x = [pool(i).squeeze(-1) for i, pool in zip(x, self.maxpools)]
        print("after pool : ", x[0].shape)
        x = torch.cat(x, 1)
        print("after cat : ", x.shape)
        x = self.classifier(x)
        print("after classifier : ", x.shape)
        return x
        

        

if __name__ == "__main__":

    vocab_dict = loadVocabDict(VOCABDICT_PATH)
    
    model = model(vocab_dict)
    sentence = "我真的很爱北京天安门"
    input = model.sentence2Input(sentence)
    model.forward(input)