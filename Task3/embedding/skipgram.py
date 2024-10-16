import MeCab
from nltk.tokenize import word_tokenize
import tqdm
import pandas as pd
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
import torch.nn.utils.rnn as rnn_utils

class SkipGramModel(nn.Module):

    """
    SkipGramModel: the skipgram model for training the word embeddings
    """

    def __init__(self, config : dict, vocab_size : int):
        super(SkipGramModel, self).__init__()
        self.assertConfig(config)
        self.embedding = nn.Embedding(vocab_size, config["embedding_dim"])
        self.output_layer = nn.Linear(config["embedding_dim"], vocab_size)

    def assertConfig(self, config):
        assert "embedding_dim" in config, "<SkipGramModel>: embedding_dim is not in config"
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.output_layer(x)
        return x
    
    def embed(self, x):
        x = self.embedding(x)
        return x

class SkipGram():

    """
    SkipGram: the class for training the skipgram model for word embeddings
    """

    def __init__(self, config : dict, data : list, jp_vocab_dict : dict, en_vocab_dict : dict):
        self.assertConfig(config)
        self.config = config
        self.data = data
        self.jp_vocab_dict = jp_vocab_dict
        self.en_vocab_dict = en_vocab_dict
        self.jp_tokenizer = MeCab.Tagger("-Owakati")

        if config['skipgram_device'] == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config['skipgram_device']

        # embedding model
        self.jp_embedding_model = SkipGramModel(config, len(jp_vocab_dict)).to(self.device)
        self.en_embedding_model = SkipGramModel(config, len(en_vocab_dict)).to(self.device)

        


    def assertConfig(self, config):

        assert "wandb_project_name" in config, "<SkipGram>: wandb_project_name is not in config"
        assert "wandb_run_name" in config, "<SkipGram>: wandb_run_name is not in config"

        assert "skipgram_window_size" in config, "<SkipGram>: skipgram_window_size is not in config"

        assert "skipgram_batch_size" in config, "<SkipGram>: skipgram_batch_size is not in config"
        assert "skipgram_lr" in config, "<SkipGram>: skipgram_lr is not in config"
        assert "skipgram_epochs" in config, "<SkipGram>: skipgram_epochs is not in config"
        assert 'skipgram_device' in config, "<SkipGram>: skipgram_device is not in config"

        assert "jp_skipgram_datset_path" in config, "<SkipGram>: jp_skipgram_datset_path is not in config"
        assert "en_skipgram_datset_path" in config, "<SkipGram>: en_skipgram_datset_path is not in config"
        assert "jp_embedding_model_path" in config, "<SkipGram>: jp_embedding_model_path is not in config"
        assert "en_embedding_model_path" in config, "<SkipGram>: en_embedding_model_path is not in config"
        
    def generateData(self) -> tuple:

        """
        generateData: generate the skipgram dataset for the Japanese and English sentences
        Parameters:
            None
        Returns:
            - (jp_pairs, en_pairs): a tuple of lists of tuples (target, context) for the Japanese and English tokenid
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
            
            # insert <START> and <END> token
            jp_indexes = [self.jp_vocab_dict["<START>"]] + jp_indexes + [self.jp_vocab_dict["<END>"]]
            en_indexes = [self.en_vocab_dict["<START>"]] + en_indexes + [self.en_vocab_dict["<END>"]]

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
            - (jp_data, en_data): a tuple of lists of tuples (input, target) for training
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
    
    def trainEmbedding(self):
        
        jp_data, en_data = self.loadSkipGramData()

        # define optimizer for each model
        jp_optimizer = torch.optim.SGD(self.jp_embedding_model.parameters(), lr=self.config["skipgram_lr"])
        en_optimizer = torch.optim.SGD(self.en_embedding_model.parameters(), lr=self.config["skipgram_lr"])

        # define criterion for each model
        jp_criterion = nn.CrossEntropyLoss()
        en_criterion = nn.CrossEntropyLoss()

        # train the embedding model
        # initialize wandb
        wandb.init(project=self.config["wandb_project_name"], name=self.config["wandb_run_name"])
        wandb.config.update(self.config, allow_val_change=True)

        # try load checkpoints
        jp_checkpoint_path = self.config["jp_embedding_model_path"]
        en_checkpoint_path = self.config["en_embedding_model_path"]
        if (os.path.exists(jp_checkpoint_path) and os.path.exists(en_checkpoint_path)):
            jp_checkpoint = torch.load(jp_checkpoint_path)
            en_checkpoint = torch.load(en_checkpoint_path)
            self.jp_embedding_model.load_state_dict(jp_checkpoint['model_state_dict'])
            self.en_embedding_model.load_state_dict(en_checkpoint['model_state_dict'])
            jp_optimizer.load_state_dict(jp_checkpoint['optimizer_state_dict'])
            en_optimizer.load_state_dict(en_checkpoint['optimizer_state_dict'])
            start_epoch = jp_checkpoint['epoch']
        else:
            start_epoch = 0

        # train loop
        for epoch in tqdm.tqdm(range(start_epoch, self.config["skipgram_epochs"])):
            for i in range(len(jp_data)):

                jp_input, jp_target = jp_data[i]
                jp_input = jp_input.to(self.device)
                jp_target = jp_target.to(self.device)

                jp_optimizer.zero_grad()
                jp_output = self.jp_embedding_model(jp_input)
                jp_loss = jp_criterion(jp_output, jp_target)
                jp_loss.backward()
                jp_optimizer.step()

                wandb.log({"jp_loss": jp_loss.item()})

                if i >= len(en_data):
                    continue

                en_input, en_target = en_data[i]
                en_input = en_input.to(self.device)
                en_target = en_target.to(self.device)

                en_optimizer.zero_grad()
                en_output = self.en_embedding_model(en_input)
                en_loss = en_criterion(en_output, en_target)
                en_loss.backward()
                en_optimizer.step()

                wandb.log({"en_loss": en_loss.item()})
                
            wandb.log({"epoch": epoch})

            # save the model checkpoints
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.jp_embedding_model.state_dict(),
                'optimizer_state_dict': jp_optimizer.state_dict()
            }, jp_checkpoint_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.en_embedding_model.state_dict(),
                'optimizer_state_dict': en_optimizer.state_dict()
            }, en_checkpoint_path)

    def embeddingSentence(self, sentences : list, lang : str) -> torch.Tensor:

        """
        embeddingSentence: embed the sentence into the word embeddings
        Parameters:
            - sentence: the sentence to be embedded
            - lang: the language of the sentence
        Returns:
            - embeddings: the word embeddings of the sentence
        """

        if lang == "jp":
            model = self.jp_embedding_model
            vocab_dict = self.jp_vocab_dict
            checkpoint_path = self.config["jp_embedding_model_path"]
            model.load_state_dict(torch.load(checkpoint_path))
            indexes_list = []
            for sentence in sentences:
                words = self.jp_tokenizer.parse(sentence).split()
                indexes = []
                for word in words:
                    if word not in vocab_dict:
                        word = "<UNK>"
                    indexes.append(vocab_dict[word])

                indexes = [vocab_dict["<START>"]] + indexes + [vocab_dict["<END>"]]
                indexes = torch.tensor(indexes)
                indexes_list.append(indexes)
        elif lang == "en":
            model = self.en_embedding_model
            vocab_dict = self.en_vocab_dict
            checkpoint_path = self.config["en_embedding_model_path"]
            model.load_state_dict(torch.load(checkpoint_path))
            indexes_list = []
            for sentence in sentences:
                words = word_tokenize(sentence)
                indexes = []
                for word in words:
                    if word not in vocab_dict:
                        word = "<UNK>"
                    indexes.append(vocab_dict[word])
                    
                indexes = [vocab_dict["<START>"]] + indexes + [vocab_dict["<END>"]]
                indexes = torch.tensor(indexes)
                indexes_list.append(indexes)
        else:
            raise ValueError("<SkipGram>: lang should be either 'jp' or 'en'")
        
        # form batch
        padded_sequences = rnn_utils.pad_sequence(indexes_list, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(seq) for seq in indexes_list])
        packed_sequence = rnn_utils.pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)
        
        # unpack the batch
        index_batch, _ = rnn_utils.pad_packed_sequence(packed_sequence, batch_first=True)

        return index_batch