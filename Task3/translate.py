
from dataloader import DataLoader
from utils import read_config
from embedding import Vocab
from embedding import SkipGram
from lstm import LSTMTrain
from nltk.tokenize import word_tokenize
import sacrebleu

if __name__ == "__main__":

    config = read_config("./config.yaml")
    
    dataloader = DataLoader(config)
    data = dataloader.loadRaw("test")
    vocab = Vocab(config, data)
    jp_vocab_dict, en_vocab_dict = vocab.loadVocab()
    skipgram = SkipGram(config, data, jp_vocab_dict, en_vocab_dict)
    # skipgram.trainEmbedding()
    # get english sentences
    lstm_train = LSTMTrain(config, en_vocab_dict, skipgram, data)

    train_data = dataloader.loadRaw("train")
    test_data = dataloader.loadRaw("test")
    val_data = dataloader.loadRaw("val")

    train_sentences = [i[0] for i in train_data][:1000]
    train_references = [[i[1] for i in train_data]][:1000]
    train_data = [train_sentences, train_references]

    test_sentences = [i[0] for i in test_data]
    test_references = [[i[1] for i in test_data]]
    test_data = [test_sentences, test_references]

    val_sentences = [i[0] for i in val_data]
    val_references = [[i[1] for i in val_data]]
    val_data = [val_sentences, val_references]

    # lstm_train.train(train_data, test_data, val_data)

    # hypothesis = lstm_train.translate(test_data[0])
    # reference = []
    # for sentence in test_data[1][0]: 
    #     reference.append(" ".join(word_tokenize(sentence)))
    # test_bleu = sacrebleu.corpus_bleu(hypothesis, [reference])
    # print(test_bleu.score)

    # perplexity = lstm_train.calPerplexity()
    # print(perplexity)

    case_1 = "私の名前は愛です"
    case_2 = "昨日はお肉を食べません"
    case_3 = "いただきますよう"
    case_4 = "秋は好きです"
    case_5 = "おはようございます"

    japanese_sentences = [case_1, case_2, case_3, case_4, case_5]
    result = lstm_train.translate(japanese_sentences)
    print()

    for i in range(len(result)):
        print(f"Japanese: {japanese_sentences[i]}")
        print(f"English: {result[i]}")
        print("\n")