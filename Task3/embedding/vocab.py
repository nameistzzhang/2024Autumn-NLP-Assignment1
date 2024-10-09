import MeCab
from nltk.tokenize import word_tokenize
import pandas as pd
import csv
import tqdm
from collections import Counter

class Vocab():
    
    """
    Vocab: a class for building the vocabularies for the Japanese and English sentences and also load vocabulary dictionary for further use
    """

    def __init__(self, config : dict, data : list):
        self.assertConfig(config)
        self.config = config
        self.data = data
        self.jp_tokenizer = MeCab.Tagger("-Owakati")

    def assertConfig(self, config : dict):
        assert "jp_vocab_path" in config, "<Vocab>: vocab_path is not in config"
        assert "en_vocab_path" in config, "<Vocab>: vocab_path is not in config"

    def loadVocab(self) -> tuple :

        jp_vocab_dict = {}
        en_vocab_dict = {}

        print("\n>>>  loading the vocab_dict from csv ... ...")
        with open(self.config['jp_vocab_path'], 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                token, idx = row
                jp_vocab_dict[token] = int(idx)

        with open(self.config['en_vocab_path'], 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                token, idx = row
                en_vocab_dict[token] = int(idx)

        return (jp_vocab_dict, en_vocab_dict)

    def buildDict(self) -> tuple :

        """
        buildDict: build the vocabularies for the Japanese and English sentences
        Parameters:
            None
        Returns:
            - (jp_vocab_dict, en_vocab_dict): a tuple of dictionaries for the Japanese and English vocabularies
        """

        en_vocab = []
        jp_vocab = []

        datalen = len(self.data)
        print("\n>>>  extracting the vocabularies ... ...")
        for i in tqdm.tqdm(range(datalen)):
            jp_sentence = self.data[i][0]
            en_sentence = self.data[i][1]

            # use MeCab to tokenize the japanese sentence
            jp_tokens = self.jp_tokenizer.parse(jp_sentence).split()
            en_tokens = word_tokenize(en_sentence)
            
            # add the tokens to the vocab
            jp_vocab.extend(jp_tokens)
            en_vocab.extend(en_tokens)

        print("Number of Japanese words: ", len(jp_vocab))
        print("Number of English words: ", len(en_vocab))


        print("\n>>> building the dictionary ... ...")
        # put the vocab into a dictionary of word:token_num
        jp_token_counts = Counter(jp_vocab)
        en_token_counts = Counter(en_vocab)

        jp_vocab_dict = {token: (idx+1) for idx, (token, _) in enumerate(jp_token_counts.items())}
        en_vocab_dict = {token: (idx+1) for idx, (token, _) in enumerate(en_token_counts.items())}

        # add the <UNK> token to the vocab
        jp_vocab_dict["<UNK>"] = len(jp_vocab_dict) + 1
        en_vocab_dict["<UNK>"] = len(en_vocab_dict) + 1

        print("Number of Japanese tokens: ", len(jp_vocab_dict))
        print("Number of English tokens: ", len(en_vocab_dict))

        # save the vocab to the file
        print("\n>>>  saving the vocab to csv ... ...")
        jp_dataframe = pd.DataFrame(jp_vocab_dict.items(), columns=['token', 'idx'])
        en_dataframe = pd.DataFrame(en_vocab_dict.items(), columns=['token', 'idx'])
        jp_dataframe.to_csv(self.config['jp_vocab_path'], index=False)
        en_dataframe.to_csv(self.config['en_vocab_path'], index=False)

        return (jp_vocab_dict, en_vocab_dict)