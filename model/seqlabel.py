# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-09-20
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from model.wordsequence import WordSequence
from model.crf import CRF

head_path, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))


class SeqLabel(nn.Module):
	def __init__(self, data):
		super(SeqLabel, self).__init__()
		label_size = data.label_alphabet_size
		data.label_alphabet_size += 2
		self.word_hidden = WordSequence(data)
		self.crf = CRF(label_size, data.gpu)

	def calculate_loss(
			self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
		outs = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
		batch_size = word_inputs.size(0)
		seq_len = word_inputs.size(1)
		total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
		scores, tag_seq = self.crf._viterbi_decode(outs, mask)
		return total_loss, tag_seq

	def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
		outs = self.word_hidden(
			word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
		batch_size = word_inputs.size(0)
		seq_len = word_inputs.size(1)
		scores, tag_seq = self.crf._viterbi_decode(outs, mask)
		return tag_seq
