import math
import torch 
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__
        
        self.d_model = d_model # d_model = 512
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, main_input):
        return self.embedding(main_input) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_length_of_seq: int, dropout: float):
        super().__init__
        self.d_model = d_model
        self.max_length_of_seq = max_length_of_seq
        self.dropout = nn.Dropout(p=dropout) # most probably 0.5

        pe = torch.zeros(max_length_of_seq, d_model) # Create Matrix [max_len x d_model]
        # arange creates a [1 x max_len] array
        pos = torch.arange(0, max_length_of_seq).unsqueeze(1) # unsqueeze creates a [max_len x 1] array
        denominator = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # both even and odd denominators are same 
        pe[:, 0::2] = torch.sin(pos * denominator)
        pe[:, 1::2] = torch.cos(pos * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # register as part of the model

    def forward(self, main_input):
        # main_input = [batch_size x seq_len x d_model]
        # pe = [batch_size (1) x max_seq_len x d_model]
        # we want our input to map to the seq_len hence size(1) <-- second index of size
        main_input = main_input + (self.pe[:, main_input.size(1), :]).requires_grad(False)
        return self.dropout(main_input)


class LayerNormalization(nn.Module):
    """
    -- Basic Concept -- 
    -> We apply LayerNormalization since we want to normalize the features 
       of each token. 
    -> Therefore, calculations are done on the last dimension of the input [batch_size x seq_len x d_model]
    -> The reason seq_len is not normalized is that it will mix the features across different tokens 
    -> batch_size can't be normalized since that is BatchNormalization ~ we want tokens to be normalized 
       based on their own features
    """

    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon # epsilon makes sure denominator never equals 0
        # Used for amplifying 
        self.alpha = nn.Parameter(torch.ones(1)) # multiplicable 
        self.gamma = nn.Parameter(torch.zeros(1)) # additive

    def forward(self, main_input):
        mean = main_input.mean(dim = -1, keepdim=True)
        std = main_input.std(dim = -1, keepdim=True) 
        return self.alpha * (main_input - mean) / (std + self.epsilon) + self.gamma

class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_w1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_w2 = nn.Linear(d_ff, d_model)

    def forward(self, main_input):
        layer1 = self.linear_w1(main_input)
        regularizer1 = self.dropout(torch.relu(layer1))
        return self.linear_w2(regularizer1)

        