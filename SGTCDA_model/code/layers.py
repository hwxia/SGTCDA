import torch
import torch.nn as nn

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    # input_dim=128 name='gan_decoder' num_d=218
    def __init__(self, input_dim, name, num_d, dropout=0., act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.num_d = num_d
        self.w = nn.Linear(input_dim * 2, input_dim * 2)
        self.w1 = nn.Linear(input_dim, input_dim)
        self.att_drug = nn.Parameter(torch.rand(2), requires_grad=True)
        self.att_cir = nn.Parameter(torch.rand(2), requires_grad=True)
        nn.init.xavier_uniform_(self.w1.weight)

    def forward(self, inputs, embd_cir, embd_drug):
        inputs = self.dropout(inputs)
        embd_drug = self.dropout(embd_drug)
        embd_cir = self.dropout(embd_cir)
        R = inputs[0:self.num_d, :]
        D = inputs[self.num_d:, :]
        R=torch.cat((R,embd_drug),1)
        D=torch.cat((D, embd_cir), 1)
        D = D.T
        x = R@D
        x = torch.reshape(x, [-1])
        outputs = self.act(x)
        return outputs



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        # Perform linear transformation and split into heads
        query = self.q_linear(query).view(-1, self.n_heads, self.head_dim)
        key = self.k_linear(key).view(-1, self.n_heads, self.head_dim)
        value = self.v_linear(value).view(-1, self.n_heads, self.head_dim)

        # Transpose to get dimensions batch_size x n_heads x seq_len x head_dim
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attention = torch.nn.functional.softmax(scores, dim=-1)

        # Multiply attention scores with value
        out = torch.matmul(attention, value)

        # Transpose back and concatenate heads
        out = out.transpose(1, 2).contiguous().view(-1, self.d_model)

        # Linear layer for final output
        out = self.out_linear(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # Multi-head attention
        attention_out = self.multihead_attention(x, x, x, mask)

        # Residual connection and normalization
        x = x + attention_out
        x = self.norm(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
