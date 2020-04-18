# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-09-23
"""

import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature  # 公式中的分母部分：根号d_k
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))  # (n_head*64, len_seq, len_seq)
        attn = attn / self.temperature  # self.temperature = 根号d_k, attn维度不变
        # slf_attn过程中每个时序只看一个字
        if mask is not None:  # (512,len(tgt),len(tgt)
            attn = attn.masked_fill(mask, -np.inf)  # mask为1的部分置为-inf，-inf就是每次都mask掉的部分

        attn = self.softmax(attn)
        attn = self.dropout(attn)  # (n_head*64,len_seq, len_seq)
        output = torch.bmm(attn, v)  # (n_head*64, len_seq, d_v)

        return output, attn
