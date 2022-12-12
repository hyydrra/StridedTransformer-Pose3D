import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import os
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N, length, d_model):
        super(Encoder, self).__init__()
        self.layers = layer
        self.norm = LayerNorm(d_model)

        self.pos_embedding_1 = nn.Parameter(torch.randn(1, length, d_model))
        self.pos_embedding_2 = nn.Parameter(torch.randn(1, length, d_model))
        self.pos_embedding_3 = nn.Parameter(torch.randn(1, length, d_model))

    def forward(self, x, mask):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x += self.pos_embedding_1[:, :x.shape[1]]
            elif i == 1:
                x += self.pos_embedding_2[:, :x.shape[1]]
            elif i == 2:
                x += self.pos_embedding_3[:, :x.shape[1]]

            x = layer(x, mask, i)

        return x

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout, stride_num, i):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.MaxPool1d(1, stride_num[i])
        
    def forward(self, x, sublayer, i=-1, stride_num=-1):
        if i != -1:
            if stride_num[i] != 1:
                res = self.pooling(x.permute(0, 2, 1))
                res = res.permute(0, 2, 1)
                
                return res + self.dropout(sublayer(self.norm(x)))
            else:
                return x + self.dropout(sublayer(self.norm(x)))
        else:
            return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, stride_num, i):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.stride_num = stride_num
        self.sublayer = clones(SublayerConnection(size, dropout, stride_num, i), 2)
        self.size = size

    def forward(self, x, mask, i):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward, i, self.stride_num)
        return x



# class MultiHeadedAttention(nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         super(MultiHeadedAttention, self).__init__()
#         assert d_model % h == 0
#         self.d_k = d_model // h 
#         self.h = h
#         self.linears = clones(nn.Linear(d_model, d_model), 4)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, query, key, value, mask=None):
#         if mask is not None:
#             mask = mask.unsqueeze(1)
#         nbatches = query.size(0)

#         query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#              for l, x in zip(self.linears, (query, key, value))]

#         x, self.attn = attention(query, key, value, mask=mask,
#                                  dropout=self.dropout)

#         x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
#         return self.linears[-1](x)


class MultiHeadedAttention(nn.Module):
# strided:
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        h (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """
    def __init__(self, h, d_model, dropout_p=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model % h should be zero."
        self.d_model = d_model
        self.d_k = int(d_model / h)
        self.h = h
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.v_bias = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            query,
            key,
            value,
            pos_embedding,
            mask= None):
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.h, self.d_k)
        key = self.key_proj(key).view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.h, self.d_k)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), key.permute(0, 2, 3, 1))
        pos_score = self._compute_relative_positional_encoding(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _compute_relative_positional_encoding(self, pos_score):
        batch_size, h, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, h, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, h, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score




class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, number = -1, stride_num=-1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_model, d_ff, kernel_size=1, stride=1)
        self.w_2 = nn.Conv1d(d_ff, d_model, kernel_size=3, stride=stride_num[number], padding = 1)

        self.gelu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.w_2(self.dropout(self.gelu(self.w_1(x))))
        x = x.permute(0, 2, 1)

        return x

class Transformer(nn.Module):   
    def __init__(self, n_layers=3, d_model=256, d_ff=512, h=8, length=27, stride_num=None, dropout=0.1):
        super(Transformer, self).__init__()

        self.length = length

        self.stride_num = stride_num
        self.model = self.make_model(N=n_layers, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout, length = self.length)

    def forward(self, x, mask=None):
        x = self.model(x, mask)

        return x

    def make_model(self, N=3, d_model=256, d_ff=512, h=8, dropout=0.1, length=27):
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)

        model_EncoderLayer = []
        for i in range(N):
            ff = PositionwiseFeedForward(d_model, d_ff, dropout, i, self.stride_num)
            model_EncoderLayer.append(EncoderLayer(d_model, c(attn), c(ff), dropout, self.stride_num, i))

        model_EncoderLayer = nn.ModuleList(model_EncoderLayer)

        model = Encoder(model_EncoderLayer, N, length, d_model)
        
        return model







