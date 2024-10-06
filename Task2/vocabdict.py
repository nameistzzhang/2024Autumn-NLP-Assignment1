from collections import Counter
import pandas as pd
import csv
import jieba

class VocabDict(): 

    """
    VocabDict: A class to build and load the vocabulary dictionary.
    """

    def __init__(self) -> None:
        pass
    
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
        
        vocabulary = {token: (idx+1) for idx, (token, _) in enumerate(token_counts.items())}
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
