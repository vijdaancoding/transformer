import math
import torch 
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # d_model = 512
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, main_input):
        return self.embedding(main_input) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_length_of_seq: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.max_length_of_seq = max_length_of_seq
        self.dropout = nn.Dropout(p=dropout) # most probably 0.5

        pe = torch.zeros(max_length_of_seq, d_model) # Create Matrix [max_len x d_model]
        # arange creates a [1 x max_len] array
        pos = torch.arange(0, max_length_of_seq).unsqueeze(1) # unsqueeze creates a [max_len x 1] array
        denominator = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # both even and odd denominators are same 
        pe[:, 0::2] = torch.sin(pos * denominator)
        pe[:, 1::2] = torch.cos(pos * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # register as part of the model

    def forward(self, main_input):
        # main_input = [batch_size x seq_len x d_model]
        # pe = [batch_size (1) x max_seq_len x d_model]
        # we want our input to map to the seq_len hence shape(1) <-- second index of size
        main_input = main_input + (self.pe[:, :main_input.shape[1], :]).requires_grad_(False)
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

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "-- d_model should be divisible by h -- \nwhere:\nd_model: # of features per token\nh: # of heads\n)"

        self.d_k = d_model // h 

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod # do not need instance of class to call the function
    def scaled_dot_product_attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # query size: [batch, h, seq_len, d_k]
        # key_transpose size: [batch, h, d_k, seq_len]
        # matmul result size: [batch, h, seq_len, seq_len]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # mask size: [batch, h, seq_len, seq_len]
        # after mask applied size: [batch, h, seq_len, seq_len]
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        # softmax applied on last seq_len 
        attention_scores = attention_scores.softmax(dim = -1)
        # after dropout size: [batch, h, seq_len, seq_len] 
        if dropout is not None: 
            attention_scores = dropout(attention_scores)

        # attention_score size: [batch, h, seq_len, seq_len]
        # value size: [batch, h, seq_len, d_v]
        # matmul result size: [batch, h, seq_len, d_v]
        return (attention_scores @ value), attention_scores

    def forward(self, q, v, k, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Splitting Q, K, V matrices into smaller matrices 
        # [Batch, Seq_len , d_model] --> [Batch, Seq_len, h, d_k] --> [Batch, h, Seq_len, d_k] 
        # Transposing to group together Seq_len and d_k since each sentence will be with its features
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        main_input, self.attention_scores = MultiHeadAttention.scaled_dot_product_attention(query, key, value, mask, self.dropout)

        # [Batch, h, seq_len, d_k] --> [batch, seq_len, h, d_k]
        main_input = main_input.transpose(1, 2).contiguous().view(main_input.shape[0], -1, self.h * self.d_k)

        return self.w_o(main_input)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, main_input, sub_layer):
        return main_input + self.dropout(sub_layer(self.norm(main_input)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        # Create a List of Modules
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, main_input, encoder_mask): 
        main_input = self.residual_connection[0](main_input, lambda x: self.self_attention(x, x, x, encoder_mask))
        main_input = self.residual_connection[1](main_input, self.feed_forward)
        return main_input

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, main_input, mask):
        for layer in self.layers:
            main_input = layer(main_input, mask)
        return self.norm(main_input)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, decoder_input, encoder_output, encoder_mask, decoder_mask):
        decoder_input = self.residual_connection[0](decoder_input, lambda x: self.self_attention(x, x, x, decoder_mask))
        decoder_input = self.residual_connection[1](decoder_input, lambda x: self.cross_attention(decoder_input, encoder_output, encoder_output, encoder_mask))
        decoder_input = self.residual_connection[2](decoder_input, self.feed_forward)

        return decoder_input

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, decoder_input, encoder_output, encoder_mask, decoder_mask):
        for layer in self.layers:
            decoder_input = layer(decoder_input, encoder_output, encoder_mask, decoder_mask)
        return self.norm(decoder_input)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, decoder_output): 
        # [batch, seq_len, d_model] --> [batch, seq_len, vocab_size]
        # Uses log_sofmax to express likelihood of each token in the vocab
        return torch.log_softmax(self.proj(decoder_output), dim=-1)

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, trgt_embedding: InputEmbeddings, src_pos: PositionalEncoding, trgt_pos: PositionalEncoding, proj_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.trgt_embedding = trgt_embedding
        self.src_pos = src_pos
        self.trgt_pos = trgt_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trgt, trgt_mask):
        trgt = self.trgt_embedding(trgt)
        trgt = self.trgt_pos(trgt)
        return self.decoder(trgt, encoder_output, src_mask, trgt_mask)

    def projection(self, decoder_output): 
        return self.proj_layer(decoder_output)

   
def build_transformer(src_vocab_size: int, trgt_vocab_size: int, src_seq_len: int, trgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # Create Embedding 
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    trgt_embedding = InputEmbeddings(d_model, trgt_vocab_size)

    # Create Position Encoders
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trgt_pos = PositionalEncoding(d_model, trgt_seq_len, dropout)

    encoder_stack = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_forward, dropout)
        encoder_stack.append(encoder_block)

    decoder_stack = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
        decoder_stack.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_stack))
    decoder = Decoder(nn.ModuleList(decoder_stack))

    proj_layer = ProjectionLayer(d_model, trgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embedding, trgt_embedding, src_pos, trgt_pos, proj_layer)

    for p in transformer.parameters():
        if p.dim() > 1: 
            nn.init.xavier_uniform_(p)

    return transformer
