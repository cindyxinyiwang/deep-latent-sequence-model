#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import sys

import numpy as np

import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import init_param


class PositionalEmbedding(nn.Module):
  def __init__(self, hparams):
    super(PositionalEmbedding, self).__init__()

    self.hparams = hparams

    if self.hparams.pos_emb_size is not None:
      self.emb = nn.Embedding(self.hparams.pos_emb_size,
                              self.hparams.d_word_vec,
                              padding_idx=0)
      if self.hparams.cuda:
        self.emb = self.emb.cuda()
    else:
      d_word_vec = self.hparams.d_word_vec
      self.emb_scale = self.hparams.init_range * d_word_vec
      freq = torch.arange(0, d_word_vec, 2).float() / d_word_vec
      self.freq = 1.0 / (10000.0 ** Variable(freq))
      #self.freq = 10000.0 ** Variable(freq)
      #print(self.freq)
      if self.hparams.cuda:
        self.freq = self.freq.cuda()

  def forward(self, x=None, pos=None):
    """Compute positional embeddings.

    Args:
      x: Tensor of size [batch_size, max_len]

    Returns:
      emb: Tensor of size [batch_size, max_len, d_word_vec].
    """

    d_word_vec = self.hparams.d_word_vec
    if pos is not None:
      batch_size, max_len = pos.size()
      pos = Variable(pos)
    else:
      batch_size, max_len = x.size()
      pos = Variable(torch.arange(0, max_len))
    if self.hparams.cuda:
      pos = pos.cuda()
    if self.hparams.pos_emb_size is not None:
      pos = pos.add_(1).long().unsqueeze(0).expand_as(x).contiguous()
      emb = self.emb(pos)
    else:
      emb = pos.float().unsqueeze(-1) * self.freq.unsqueeze(0)
      sin = torch.sin(emb).mul_(self.emb_scale).unsqueeze(-1)
      cos = torch.cos(emb).mul_(self.emb_scale).unsqueeze(-1)
      #emb = pos.float().unsqueeze(-1) / self.freq.unsqueeze(0)
      #sin = torch.sin(emb).unsqueeze(-1)
      #cos = torch.cos(emb).unsqueeze(-1)
      emb = torch.cat([sin, cos], dim=-1).contiguous().view(max_len, d_word_vec)
      emb = emb.unsqueeze(0).expand(batch_size, -1, -1)

    return emb


class LayerNormalization(nn.Module):
  def __init__(self, d_hid, hparams, eps=1e-9):
    super(LayerNormalization, self).__init__()

    self.d_hid = d_hid
    if hasattr(hparams, "layernorm_eps"):
      self.eps = hparams.layernorm_eps
    else:
      self.eps = eps
    self.scale = nn.Parameter(torch.ones(self.d_hid), requires_grad=True)
    self.offset= nn.Parameter(torch.zeros(self.d_hid), requires_grad=True)

  def forward(self, x):
    assert x.dim() >= 2
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return self.scale * (x - mean) / (std + self.eps) + self.offset


class ScaledDotProdAttn(nn.Module):
  def __init__(self, hparams):
    super(ScaledDotProdAttn, self).__init__()
    self.temp = np.power(hparams.d_model, 0.5)
    self.dropout = nn.Dropout(hparams.dropout)
    self.softmax = nn.Softmax(dim=2)
    self.hparams = hparams

  def forward(self, q, k, v, attn_mask=None):
    """Compute Softmax(q * k.T / sqrt(dim)) * v

    Args:
      q: [batch_size, len_q, d_q].
      k: [batch_size, len_k, d_k].
      v: [batch_size, len_v, d_v].
    
    Note: batch_size may be n_heads * batch_size, but we don't care.
    
    Must have:
      d_q == d_k
      len_k == len_v

    Returns:
      attn: [batch_size, len_q, d_v].
    """
    if q.dim() == 4:
      # batch_size, n_heads, len, dim
      batch_q, len_q, d_q, n_heads = q.size()
      batch_k, len_k, d_k, n_heads = k.size()
      batch_v, len_v, d_v, n_heads = v.size()
    
      # batch_size, len_q, len_k, n_heads
      attn = torch.einsum("bidn,bjdn->bijn", (q, k)) / self.temp
      # attn_mask: [batch_size, len_q, len_k]
      if attn_mask is not None:
        attn.data.masked_fill_(attn_mask.unsqueeze(3), -self.hparams.inf)
      attn = self.softmax(attn).contiguous()
      attn = self.dropout(attn)
      output = torch.einsum("bijn,bjdn->bidn", (attn, v)).contiguous().view(batch_q, len_q, -1)
      return output

    batch_q, len_q, d_q = q.size()
    batch_k, len_k, d_k = k.size()
    batch_v, len_v, d_v = v.size()

    assert batch_q == batch_k and batch_q == batch_v
    assert d_q == d_k and len_k == len_v

    # [batch_size, len_q, len_k]
    attn = torch.bmm(q, k.transpose(1, 2)) / self.temp

    # attn_mask: [batch_size, len_q, len_k]
    if attn_mask is not None:
      #attn.data.masked_fill_(attn_mask, -float("inf"))
      attn.data.masked_fill_(attn_mask, -self.hparams.inf)
    size = attn.size()
    assert len(size) > 2 and len_q == size[1] and len_k == size[2]

    # softmax along the len_k dimension
    # [batch_size, len_q, len_k]
    attn = self.softmax(attn).contiguous()

    # [batch_size, len_q, len_k == len_v]
    attn = self.dropout(attn)

    # [batch_size, len_q, d_v]
    output = torch.bmm(attn, v).contiguous()

    return output


class RelativeMultiHeadAttn(nn.Module):
  def __init__(self, hparams):
    super(RelativeMultiHeadAttn, self).__init__()

    self.hparams = hparams

    self.attention = ScaledDotProdAttn(hparams)
    self.layer_norm = LayerNormalization(hparams.d_model, hparams)
    self.temp = np.power(hparams.d_model, 0.5)
    self.softmax = nn.Softmax(dim=-1)
    self.pos_emb = PositionalEmbedding(hparams)
    self.dropout = nn.Dropout(hparams.dropout)
    # projection of concatenated attn
    n_heads = self.hparams.n_heads
    d_model = self.hparams.d_model
    d_q = self.hparams.d_k
    d_k = self.hparams.d_k
    d_v = self.hparams.d_v

    #self.q = nn.Linear(d_model, n_heads * d_q, bias=False)
    #self.k = nn.Linear(d_model, n_heads * d_k, bias=False)
    #self.v = nn.Linear(d_model, n_heads * d_v, bias=False)
    #self.r = nn.Linear(d_model, n_heads * d_v, bias=False)
    #init_param(self.q.weight, init_type="uniform", init_range=hparams.init_range)
    #init_param(self.k.weight, init_type="uniform", init_range=hparams.init_range)
    #init_param(self.v.weight, init_type="uniform", init_range=hparams.init_range)
    #init_param(self.r.weight, init_type="uniform", init_range=hparams.init_range)

    Q, K, V, R = [], [], [], []
    for head_id in range(n_heads):
      q = nn.Linear(d_model, d_q, bias=False)
      k = nn.Linear(d_model, d_k, bias=False)
      v = nn.Linear(d_model, d_v, bias=False)
      r = nn.Linear(self.hparams.d_word_vec, d_k, bias=False)
      init_param(q.weight, init_type="uniform", init_range=hparams.init_range)
      init_param(k.weight, init_type="uniform", init_range=hparams.init_range)
      init_param(v.weight, init_type="uniform", init_range=hparams.init_range)
      init_param(r.weight, init_type="uniform", init_range=hparams.init_range)
      Q.append(q)
      K.append(k)
      V.append(v)
      R.append(r)
    self.Q = nn.ModuleList(Q)
    self.K = nn.ModuleList(K)
    self.V = nn.ModuleList(V)
    self.R = nn.ModuleList(R)
    if self.hparams.cuda:
      self.Q = self.Q.cuda()
      self.K = self.K.cuda()
      self.V = self.V.cuda()
      self.R = self.R.cuda()
      #self.q = self.q.cuda()
      #self.k = self.k.cuda()
      #self.r = self.r.cuda()
    if self.hparams.relative_pos_c:
      #self.u = nn.Linear(1, d_q, bias=False)
      self.u = nn.Linear(d_q, 1, bias=False)
      init_param(self.u.weight, init_type="uniform", init_range=hparams.init_range)
    if self.hparams.relative_pos_d:
      #self.v = nn.Linear(1, d_q, bias=False)
      self.v = nn.Linear(d_q, 1, bias=False)
      init_param(self.v.weight, init_type="uniform", init_range=hparams.init_range)
    self.w_proj = nn.Linear(n_heads * d_v, d_model, bias=False)
    init_param(self.w_proj.weight, init_type="uniform", init_range=hparams.init_range)
    if self.hparams.cuda:
      self.w_proj = self.w_proj.cuda()
      if self.hparams.relative_pos_c:
        self.u = self.u.cuda()
      if self.hparams.relative_pos_d:
        self.v = self.v.cuda()

  def forward(self, q, k, v, attn_mask=None):
    """Performs the following computations:

         head[i] = Attention(q * w_q[i], k * w_k[i], v * w_v[i])
         outputs = concat(all head[i]) * self.w_proj

    Args:
      q: [batch_size, len_q, d_q].
      k: [batch_size, len_k, d_k].
      v: [batch_size, len_v, d_v].

    Must have: len_k == len_v
    Note: This batch_size is in general NOT the training batch_size, as
      both sentences and time steps are batched together for efficiency.

    Returns:
      outputs: [batch_size, len_q, d_model].
    """

    residual = q 

    n_heads = self.hparams.n_heads
    d_model = self.hparams.d_model
    d_q = self.hparams.d_k
    d_k = self.hparams.d_k
    d_v = self.hparams.d_v
    batch_size = q.size(0)

    #r = torch.arange(len_q, -len_k, -1).unsqueeze(0)
    ## [1, len_q + len_k, d_word_vec]
    #r = self.pos_emb(pos=r)

    ## batch_size, len, d_q * n_head
    #head_q, head_k, head_v, head_r = self.q(q), self.k(k), self.v(v), self.r(r)
    ## batch_size, len, dim, n_head
    #head_q = head_q.view(batch_size, q.size(1), -1, n_heads)
    #head_k = head_k.view(batch_size, k.size(1), -1, n_heads)
    #head_v = head_v.view(batch_size, v.size(1), -1, n_heads)

    ## batch_size, len_q, len_k, n_heads
    #attn = torch.einsum("bidn,bjdn->bijn", (head_q, head_k)) / self.temp
    ## attn_mask: [batch_size, len_q, len_k]
    #if attn_mask is not None:
    #  attn.data.masked_fill_(attn_mask.unsqueeze(3), -self.hparams.inf)
    #attn = self.softmax(attn).contiguous()
    #attn = self.dropout(attn)
    #output = torch.einsum("bijn,bjdn->bidn", (attn, v)).contiguous().view(batch_q, len_q, -1)

    heads = []
    for Q, K, V, R in zip(self.Q, self.K, self.V, self.R):

      head_q, head_k, head_v = Q(q), K(k), V(v)
      batch_size, len_q, d_q = head_q.size()
      batch_size, len_k, d_k = head_k.size()
      batch_size, len_v, d_v = head_v.size()
      assert d_q == d_k and len_k == len_v

      relative_pos = torch.arange(len_q, -len_k, -1).unsqueeze(0)
      # [len_q + len_k, d_word_vec]
      relative_pos_emb = self.pos_emb(pos=relative_pos).squeeze(0)
      # [len_q + len_k, d_model]
      head_r = R(relative_pos_emb)


      # [batch_size, len_q, len_k]
      #attn = torch.bmm(head_q+self.u.weight.view(1, d_q), head_k.transpose(1, 2)) / self.temp
      attn_a = torch.bmm(head_q, head_k.transpose(1, 2))
      if self.hparams.relative_pos_c:
        # [batch_size, len_k, 1]
        attn_c = self.u(head_k).transpose(1, 2) 
        attn = (attn_a + attn_c)
      else:
        attn = attn_a
      #attn = torch.bmm(head_q, head_k.transpose(1, 2)) / self.temp

      # [batch_size, len_q, len_q + len_k]
      #attn_pos = (head_q+self.v.weight.view(1, d_q)).matmul(head_r.transpose(0, 1)) / self.temp
      attn_pos_b = (head_q).matmul(head_r.transpose(0, 1)) 
      if self.hparams.relative_pos_d:
        # [len_q + len_k, 1]
        attn_pos_d = self.v(head_r).view(1, 1, -1)
        attn_pos = (attn_pos_b + attn_pos_d)
      else:
        attn_pos = attn_pos_b 
      batch_pos_emb = []
      for i in range(len_q):
        # [batch_size, 1, len_k]
        batch_pos_emb.append(attn_pos[:,i,len_q-i:len_q+len_k-i])
        #print(batch_pos_emb[-1].size())
      attn_pos = torch.stack(batch_pos_emb, dim=1)
      attn = (attn + attn_pos) / self.temp
      # attn_mask: [batch_size, len_q, len_k]
      if attn_mask is not None:
        attn.data.masked_fill_(attn_mask, -self.hparams.inf)
      size = attn.size()
      assert len(size) > 2 and len_q == size[1] and len_k == size[2]

      # softmax along the len_k dimension
      # [batch_size, len_q, len_k]
      attn = self.softmax(attn).contiguous()

      # [batch_size, len_q, len_k == len_v]
      attn = self.dropout(attn)

      # [batch_size, len_q, d_v]
      head = torch.bmm(attn, head_v).contiguous()

      heads.append(head)

    outputs = torch.cat(heads, dim=-1).contiguous().view(batch_size, -1, n_heads * d_v)
    outputs = self.w_proj(outputs)
    outputs = self.layer_norm(outputs + residual)

    return outputs


class MultiHeadAttn(nn.Module):
  def __init__(self, hparams):
    super(MultiHeadAttn, self).__init__()

    self.hparams = hparams

    self.attention = ScaledDotProdAttn(hparams)
    self.layer_norm = LayerNormalization(hparams.d_model, hparams)

    # projection of concatenated attn
    n_heads = self.hparams.n_heads
    d_model = self.hparams.d_model
    d_q = self.hparams.d_k
    d_k = self.hparams.d_k
    d_v = self.hparams.d_v
    # d_q == d_k == k_v
    self.q = nn.Linear(d_model, n_heads * d_q, bias=False)
    self.k = nn.Linear(d_model, n_heads * d_k, bias=False)
    self.v = nn.Linear(d_model, n_heads * d_v, bias=False)
    init_param(self.q.weight, init_type="uniform", init_range=hparams.init_range)
    init_param(self.k.weight, init_type="uniform", init_range=hparams.init_range)
    init_param(self.v.weight, init_type="uniform", init_range=hparams.init_range)

    # Q, K, V = [], [], []
    # for head_id in range(n_heads):
    #   q = nn.Linear(d_model, d_q, bias=False)
    #   k = nn.Linear(d_model, d_k, bias=False)
    #   v = nn.Linear(d_model, d_v, bias=False)
    #   init_param(q.weight, init_type="uniform", init_range=hparams.init_range)
    #   init_param(k.weight, init_type="uniform", init_range=hparams.init_range)
    #   init_param(v.weight, init_type="uniform", init_range=hparams.init_range)
    #   Q.append(q)
    #   K.append(k)
    #   V.append(v)
    # self.Q = nn.ModuleList(Q)
    # self.K = nn.ModuleList(K)
    # self.V = nn.ModuleList(V)
    if self.hparams.cuda:
      #self.Q = self.Q.cuda()
      #self.K = self.K.cuda()
      #self.V = self.V.cuda()
      self.q = self.q.cuda()
      self.k = self.k.cuda()
      self.v = self.v.cuda()

    self.w_proj = nn.Linear(n_heads * d_v, d_model, bias=False)
    init_param(self.w_proj.weight, init_type="uniform", init_range=hparams.init_range)
    if self.hparams.cuda:
      self.w_proj = self.w_proj.cuda()

  def forward(self, q, k, v, attn_mask=None):
    """Performs the following computations:

         head[i] = Attention(q * w_q[i], k * w_k[i], v * w_v[i])
         outputs = concat(all head[i]) * self.w_proj

    Args:
      q: [batch_size, len_q, d_q].
      k: [batch_size, len_k, d_k].
      v: [batch_size, len_v, d_v].

    Must have: len_k == len_v
    Note: This batch_size is in general NOT the training batch_size, as
      both sentences and time steps are batched together for efficiency.

    Returns:
      outputs: [batch_size, len_q, d_model].
    """

    residual = q 

    n_heads = self.hparams.n_heads
    d_model = self.hparams.d_model
    d_q = self.hparams.d_k
    d_k = self.hparams.d_k
    d_v = self.hparams.d_v
    batch_size = q.size(0)
    # batch_size, len, d_q * n_head
    head_q, head_k, head_v = self.q(q), self.k(k), self.v(v)
    # batch_size, len, dim, n_head
    head_q = head_q.view(batch_size, q.size(1), -1, n_heads)
    head_k = head_k.view(batch_size, k.size(1), -1, n_heads)
    head_v = head_v.view(batch_size, v.size(1), -1, n_heads)
    outputs = self.attention(head_q, head_k, head_v, attn_mask=attn_mask)
    #heads = []
    #for Q, K, V in zip(self.Q, self.K, self.V):
    #  head_q, head_k, head_v = Q(q), K(k), V(v)
    #  head = self.attention(head_q, head_k, head_v, attn_mask=attn_mask)
    #  heads.append(head)

    #outputs = torch.cat(heads, dim=-1).contiguous().view(batch_size, -1, n_heads * d_v)
    outputs = self.w_proj(outputs)
    outputs = self.layer_norm(outputs + residual)

    return outputs


class PositionwiseFF(nn.Module):
  def __init__(self, hparams):
    super(PositionwiseFF, self).__init__()
    self.hparams = hparams

    self.w_1 = nn.Linear(hparams.d_model, hparams.d_inner, bias=False)
    self.w_2 = nn.Linear(hparams.d_inner, hparams.d_model, bias=False)
    self.dropout = nn.Dropout(hparams.dropout)
    self.relu = nn.ReLU()
    self.layer_norm = LayerNormalization(hparams.d_model, hparams)

    init_param(self.w_1.weight, init_type="uniform", init_range=hparams.init_range)
    init_param(self.w_2.weight, init_type="uniform", init_range=hparams.init_range)


  def forward(self, x):
    residual = x
    batch_size, x_len, d_model = x.size()
    x = self.relu(self.w_1(x.view(-1, d_model)))
    x = self.w_2(x).view(batch_size, x_len, d_model)
    x = self.dropout(x)
    x += residual
    x = self.layer_norm(x)
    return x


class EncoderLayer(nn.Module):
  """Compose multi-head attention and positionwise feeding."""

  def __init__(self, hparams):
    super(EncoderLayer, self).__init__()

    self.hparams = hparams
    if self.hparams.transformer_relative_pos:
      self.attn = RelativeMultiHeadAttn(hparams)
    else:
      self.attn = MultiHeadAttn(hparams)
    self.pos_ff = PositionwiseFF(hparams)

  def forward(self, enc_input, attn_mask=None):
    """Normal forward pass.

    Args:
      enc_input: [batch_size, x_len, d_model].
      attn_mask: [batch_size, x_len, x_len].
    """

    enc_output = self.attn(enc_input, enc_input, enc_input, attn_mask=attn_mask)
    enc_output = self.pos_ff(enc_output)
    return enc_output


class DecoderLayer(nn.Module):
  """Multi-head attention to both input_states and output_states."""

  def __init__(self, hparams):
    super(DecoderLayer, self).__init__()
    self.hparams = hparams
    if not self.hparams.transformer_relative_pos:
      self.y_attn = MultiHeadAttn(hparams)
      self.x_attn = MultiHeadAttn(hparams)
    else:
      self.y_attn = RelativeMultiHeadAttn(hparams)
      #self.y_attn = MultiHeadAttn(hparams)
      self.x_attn = RelativeMultiHeadAttn(hparams)
      #self.x_attn = MultiHeadAttn(hparams)
    self.pos_ffn = PositionwiseFF(hparams)

  def forward(self, dec_input, enc_output, y_attn_mask=None, x_attn_mask=None, n_corrupts=0):
    """Decoder.

    Args:
      y_attn_mask: self attention mask.
      x_attn_mask: decoder-encoder attention mask.
    """

    output = self.y_attn(dec_input, dec_input, dec_input, attn_mask=y_attn_mask)
    batch_size = dec_input.size(0)
    if n_corrupts > 0:
      #print(output)
      output = output.repeat(1, 1, n_corrupts).view(batch_size*n_corrupts, -1, self.hparams.d_model)
      #print(output)
    output = self.x_attn(output, enc_output, enc_output, attn_mask=x_attn_mask)
    output = self.pos_ffn(output)
    if n_corrupts > 0:
      output = output.view(-1, n_corrupts, self.hparams.d_model)
      #print(output)
      output = torch.sum(output, dim=1).div_(n_corrupts).view(batch_size, -1, self.hparams.d_model)
      #print(output)
    return output

