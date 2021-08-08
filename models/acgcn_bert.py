# -*- coding: utf-8 -*-

import math
from os import openpty
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import final
from layers.dynamic_rnn import DynamicLSTM
from transformers import BertConfig, BertModel

class Highway(nn.Module):
    def __init__(self, layer_num, dim):
        super().__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x

class LayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_dim))
        self.beta = nn.Parameter(torch.zeros(config.hidden_dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        if config.hidden_dim % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_dim, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_dim / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_dim, self.all_head_size)
        self.key = nn.Linear(config.hidden_dim, self.all_head_size)
        self.value = nn.Linear(config.hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class ACGCN_BERT(nn.Module):
    def __init__(self, opt):
        super(ACGCN_BERT, self).__init__()
        self.opt = opt

        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout_bert = nn.Dropout(self.config.hidden_dropout_prob)
        if opt.highway:
           self.highway = Highway(opt.num_layers, self.config.hidden_size)
        
        if opt.layernorm:
            self.layernorm = LayerNorm(opt)
        self.gc1 = GraphConvolution(self.config.hidden_size, self.config.hidden_size)
        self.gc2 = GraphConvolution(self.config.hidden_size, self.config.hidden_size)
        self.MHA = SelfAttention(opt)
        self.dense = nn.Linear(2*self.config.hidden_size, self.config.hidden_size)
        self.fc = nn.Linear(self.config.hidden_size, opt.polarities_dim)

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_indices, text_trans_indices, aspect_trans_indices, left_trans_indices, adj = inputs
        # ----------Embedding Layer-----------
        outputs = self.bert(text_indices)[0]
        output = outputs[:, 1:, :]
        # ------------------------------------
        max_len=max([len(item) for item in text_trans_indices])
        text_len=torch.Tensor([len(item) for item in text_trans_indices]).long().cuda()
        aspect_len=torch.Tensor([len(item) for item in aspect_trans_indices]).long().cuda()
        left_len=torch.Tensor([len(item) for item in left_trans_indices]).long().cuda()
        tmps=torch.zeros(text_indices.size(0), max_len, self.config.hidden_size).float().cuda()
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)

        # Remove the diff brought by WordPiece Tokenization
        for i, spans in enumerate(text_trans_indices):
            for j, span in enumerate(spans):
                tmps[i,j] = torch.sum(output[i,span[0]:span[1]],0) / (span[1] - span[0])
        text_out = self.dropout_bert(tmps)

        if self.opt.highway:
            text_out = self.highway(text_out)

        # ----------Local Features------------
        if self.opt.layernorm:
            x = F.relu(self.layernorm(self.gc1(text_out, adj)))
            x = F.relu(self.layernorm(self.gc2(x, adj)))
        else:
            x = F.relu(self.gc1(text_out, adj))
            x = F.relu(self.gc2(x, adj))
        x = self.mask(x, aspect_double_idx)
        # ------------------------------------

        # ----------Get Information for Classfication-----------
        merged_features = torch.cat([text_out, x], dim=-1)  
        dense_features = self.dense(merged_features)
        mha_out = self.MHA(dense_features)
        pooled = torch.mean(mha_out, dim=1)
        output = self.fc(pooled)
        # ------------------------------------------------------
        
        return output
