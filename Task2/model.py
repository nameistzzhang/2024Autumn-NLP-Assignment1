
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    """
    Model: A class to build the CNN model for NLP task.
    """

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
        super(Model, self).__init__()

        # vocab_dict_size is the length of the vocabulary dictionary with the UNKNOWN token
        self.vocab_dict_size = len(vocab_dict) + 1

        # device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        """ 
        Model architecture: 
        1. Embedding layer: token embedding
        2. Convolutional layer: multiple filters with different window sizes
        3. Maxpooling layer: maxpooling over the convolutional layer
        4. Classifier: fully connected layers (applying dropout)
        """
        # embedding layer
        self.embedding = nn.Embedding(self.vocab_dict_size, embedding_dim).to(self.device)
        # convoluctional layer with multiple filters(filter_windows[i] * filter_num)
        self.convs = nn.ModuleList([nn.Conv2d(
            in_channels = 1, 
            out_channels=filter_num, 
            kernel_size=(window, embedding_dim))for window in filter_windows]).to(self.device)
        # maxpooling layer(conduct maxpooling on the timeseq dimension)
        self.maxpools = nn.ModuleList([nn.AdaptiveMaxPool1d(1) for _ in filter_windows]).to(self.device)
        # classifier(with dropout)
        self.classifier = nn.Sequential(
            nn.Linear(filter_num * len(filter_windows), linear_hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(linear_hidden_layers[0], linear_hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(linear_hidden_layers[1], class_num)
        ).to(self.device)

        # l2 constraint
        self.l2_constraint = l2_constraint

    def forward(self, x):

        """
        forward: Given the input tensor x, conduct the forward pass of the model.
        Parameters:
            x: input tensor with shape (batch_size, seq_len).
        """

        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(-1) for conv in self.convs]
        x = [pool(i).squeeze(-1) for i, pool in zip(x, self.maxpools)]
        x = torch.cat(x, 1)
        x = self.classifier(x)
        return x