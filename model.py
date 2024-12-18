import math
import torch 
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__
        
        self.d_model = d_model # d_model = 512
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, embedding_input):
        return self.embedding(embedding_input) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_length_of_seq: int, dropout: float):
        super().__init__
        self.d_model = d_model
        self.max_length_of_seq = max_length_of_seq
        self.dropout = nn.Dropout(p=dropout) # most probably 0.5

    def forward(self, positional_enc_input):
        pe = torch.zeros(self.max_length_of_seq)
