
import torch
import jieba

class DataLoader():

    """
    DataLoader: A class to load the dataset and convert the text to input tensor.
    """

    def __init__(self, 
        database_path : str, 
        vocab_dict : dict, 
        batch_size : int = 64,
        timeseq_len : int = 32, 
        ) -> None:

        # initialize the variables
        self.database_path = database_path
        self.vocab_dict = vocab_dict
        self.unknown_token_idx = len(vocab_dict) + 1
        self.batch_size = batch_size
        self.timeseq_len = timeseq_len

    def loadTrain(self):

        # read the dataset file
        with open(self.database_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        data = [line.strip().split('\t') for line in data]

        # convert sentence to tensor of each text and its label
        dataset = []
        for text, label in data:
            input_tensor = self.sentence2Tensor(text)
            label_tensor = torch.tensor(int(label))
            dataset.append((input_tensor, label_tensor))

        # form batches
        input_data = []
        label_data = []
        for i in range(0, len(dataset), self.batch_size):

            if i + self.batch_size > len(dataset):
                input_batch = torch.stack([data[0] for data in dataset[i:]])
                label_batch = torch.stack([data[1] for data in dataset[i:]])
            else:
                input_batch = torch.stack([data[0] for data in dataset[i:i+self.batch_size]])
                label_batch = torch.stack([data[1] for data in dataset[i:i+self.batch_size]])

            input_data.append(input_batch)
            label_data.append(label_batch)

        return input_data, label_data
    
    def loadVal(self):

        # read the dataset file
        with open(self.database_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        data = [line.strip().split('\t') for line in data]

        # convert sentence to tensor of each text and its label
        input_data = []
        label_data = []
        for text, label in data:
            input_tensor = self.sentence2Tensor(text)
            label_tensor = torch.tensor(int(label))
            input_tensor = torch.stack([input_tensor])
            label_tensor = torch.stack([label_tensor])
            input_data.append(input_tensor)
            label_data.append(label_tensor)

        return input_data, label_data
            
    def sentence2Tensor(self, sentence : str) -> torch.Tensor:

        # get a random sentence and tokenize it
        sentence = list(jieba.cut(sentence))
        sentence_idx = []
        for token in sentence:
            if token in self.vocab_dict:
                sentence_idx.append(self.vocab_dict[token])
            else:    # which means unkonwn token
                sentence_idx.append(self.unknown_token_idx - 1)
        
        # token embedding
        if len(sentence_idx) > self.timeseq_len:
            raise ValueError("The length of the sentence is larger than the timeseq_len.")
        elif len(sentence_idx) < self.timeseq_len:
            sentence_idx += [0] * (self.timeseq_len - len(sentence_idx))
        
        # convert the list to tensor
        sentence_idx = torch.tensor(sentence_idx)

        return sentence_idx