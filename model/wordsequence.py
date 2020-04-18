# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-09-20
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.wordrep import WordRep


class WordSequence(nn.Module):
	def __init__(self, data):
		super(WordSequence, self).__init__()
		self.droplstm = nn.Dropout(data.dropout)
		self.wordrep = WordRep(data)
		self.input_size = data.word_emb_dim + data.char_hidden_dim +data.feature_emb_dim
		lstm_hidden = data.hidden_dim // 2
		self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
		self.hidden2tag = nn.Linear(data.hidden_dim, data.label_alphabet_size)  # 200 -> 12

	def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
		word_represent = self.wordrep(
			word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
		packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
		hidden = None
		lstm_out, hidden = self.lstm(packed_words, hidden)
		lstm_out, _ = pad_packed_sequence(lstm_out)
		feature_out = self.droplstm(lstm_out.transpose(1,0))

		outputs = self.hidden2tag(feature_out)
		return outputs
