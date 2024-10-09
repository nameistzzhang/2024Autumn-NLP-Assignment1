
import yaml
from dataloader import DataLoader
from utils import read_config
from embedding import Vocab
from embedding import SkipGram

if __name__ == "__main__":

    config = read_config("./config.yaml")
    
    dataloader = DataLoader(config)
    data = dataloader.loadRaw("train")
    vocab = Vocab(config, data)
    jp_vocab_dict, en_vocab_dict = vocab.loadVocab()
    skipgram = SkipGram(config, data, jp_vocab_dict, en_vocab_dict)
    skipgram.loadSkipGramData()