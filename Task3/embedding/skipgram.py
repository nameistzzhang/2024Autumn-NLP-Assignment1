import MeCab
from nltk.tokenize import word_tokenize
import tqdm
import pandas as pd
import csv
import random
import torch
import torch.nn as nn
import torch.functional as F

class SkipGram():

    def __init__(self, config : dict, data : list, jp_vocab_dict : dict, en_vocab_dict : dict):
        self.assertConfig(config)
        self.config = config
        self.data = data
        self.jp_vocab_dict = jp_vocab_dict
        self.en_vocab_dict = en_vocab_dict
        self.jp_tokenizer = MeCab.Tagger("-Owakati")


    def assertConfig(self, config):
        assert "skipgram_window_size" in config, "<SkipGram>: skipgram_window_size is not in config"
        assert "skipgram_batch_size" in config, "<SkipGram>: skipgram_batch_size is not in config"
        assert "jp_skipgram_datset_path" in config, "<SkipGram>: jp_skipgram_datset_path is not in config"
        assert "en_skipgram_datset_path" in config, "<SkipGram>: en_skipgram_datset_path is not in config"
        
    def generateData(self) -> tuple:

        """
        generateData: generate the skipgram dataset for the Japanese and English sentences
        Parameters:
            None
        Returns:
            - (jp_pairs, en_pairs): a tuple of lists of tuples (target, context) for the Japanese and English sentences
        """
        
        jp_pairs = []
        en_pairs = []

        print("\n>>>  generating the skipgram dataset ... ...")
        for dataline in tqdm.tqdm(self.data):
            jp_sentence = dataline[0]
            en_sentence = dataline[1]

            # tokenize the sentence
            jp_words = self.jp_tokenizer.parse(jp_sentence).split()
            en_words = word_tokenize(en_sentence)

            # convert the words to indices
            jp_indexes = [self.jp_vocab_dict[word] for word in jp_words]
            en_indexes = [self.en_vocab_dict[word] for word in en_words]

            # generate the skipgram dataset
            for i, target_idx in enumerate(jp_indexes):
                context_start = max(0, i - self.config["skipgram_window_size"])
                context_end = min(len(jp_indexes), i + self.config["skipgram_window_size"])
                for j in range(context_start, context_end):
                    if i != j:
                        jp_pairs.append((target_idx, jp_indexes[j]))
            for i, target_idx in enumerate(en_indexes):
                context_start = max(0, i - self.config["skipgram_window_size"])
                context_end = min(len(en_indexes), i + self.config["skipgram_window_size"])
                for j in range(context_start, context_end):
                    if i != j:
                        en_pairs.append((target_idx, en_indexes[j]))

        # save the dataset to csv files
        print(">>>  saving the skipgram dataset ... ...")
        jp_dataframe = pd.DataFrame(jp_pairs, columns=["target", "context"])
        en_dataframe = pd.DataFrame(en_pairs, columns=["target", "context"])
        jp_dataframe.to_csv(self.config["jp_skipgram_datset_path"], index=False)
        en_dataframe.to_csv(self.config["en_skipgram_datset_path"], index=False)

        return (jp_pairs, en_pairs)
    
    def loadSkipGramData(self) -> tuple:

        """
        loadSkipGramData: load the skipgram dataset from the csv files into pytorch input tensors and target tensors
        Parameters:
            None
        Returns:
            - (jp_data, en_data): a tuple of lists of tuples (input, target) for the Japanese and English sentences
        """

        # load the skipgram dataset from the csv files
        jp_pairs = []
        en_pairs = []
        with open(self.config["jp_skipgram_datset_path"], "r") as f:
            reader = csv.reader(f)
            for row in reader:
                # if header then skip
                if row[0] == "target":
                    continue
                jp_pairs.append([int(row[0]), int(row[1])])
        with open(self.config["en_skipgram_datset_path"], "r") as f:
            reader = csv.reader(f)
            for row in reader:
                # if header then skip
                if row[0] == "target":
                    continue
                en_pairs.append([int(row[0]), int(row[1])])

        # shuffle the dataset
        random.shuffle(jp_pairs)
        random.shuffle(en_pairs)

        # form batches
        batchsize = self.config["skipgram_batch_size"]
        jp_data = []
        en_data = []
        print("\n>>>  forming batches for jp_data ... ...")
        for i in tqdm.tqdm(range(0, len(jp_pairs), batchsize)):
            cur_batch_input = []
            cur_batch_target = []
            for j in range(i, min(i+batchsize, len(jp_pairs))):
                cur_batch_input.append(jp_pairs[j][0])
                cur_batch_target.append(jp_pairs[j][1])
            # convert to torch tensor
            cur_batch_input = torch.tensor(cur_batch_input)
            cur_batch_target = torch.tensor(cur_batch_target)
            jp_data.append((cur_batch_input, cur_batch_target))
        
        print(">>>  forming batches for en_data ... ...")
        for i in tqdm.tqdm(range(0, len(en_pairs), batchsize)):
            cur_batch_input = []
            cur_batch_target = []
            for j in range(i, min(i+batchsize, len(en_pairs))):
                cur_batch_input.append(en_pairs[j][0])
                cur_batch_target.append(en_pairs[j][1])
            # convert to torch tensor
            cur_batch_input = torch.tensor(cur_batch_input)
            cur_batch_target = torch.tensor(cur_batch_target)
            en_data.append((cur_batch_input, cur_batch_target))

        return (jp_data, en_data)