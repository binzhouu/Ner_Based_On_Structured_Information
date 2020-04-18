# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-09-20
"""
import torch
import torch.nn as nn
import numpy as np
from model.charcnn import CharCNN


class WordRep(nn.Module):
	def __init__(self, data):
		super(WordRep, self).__init__()
		self.char_hidden_dim = data.char_hidden_dim  # 50
		self.char_embedding_dim = data.char_emb_dim  # 300
		self.char_feature = CharCNN(data.char_alphabet_size, data.pretrain_char_embedding, self.char_embedding_dim,
									self.char_hidden_dim, data.dropout)
		self.drop = nn.Dropout(data.dropout)
		self.word_embedding = nn.Embedding(data.word_alphabet_size, data.word_emb_dim)

		self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(
			data.word_alphabet_size, data.word_emb_dim)))
		self.feature_embedding = nn.Embedding(data.feat_alphabet_size, data.feature_emb_dim)
		self.feature_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_feature_embeddings))

	def random_embedding(self, vocab_size, embedding_dim):
		pretrain_emb = np.empty([vocab_size, embedding_dim])
		scale = np.sqrt(3.0 / embedding_dim)
		for index in range(vocab_size):
			pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
		return pretrain_emb

	def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
		batch_size = word_inputs.size(0)
		sent_len = word_inputs.size(1)
		word_embs = self.word_embedding(word_inputs)
		word_list = [word_embs]
		word_list.append(self.feature_embedding(feature_inputs))

		char_features = self.char_feature.get_last_hiddens(char_inputs)
		char_features = char_features[char_seq_recover]
		char_features = char_features.view(batch_size, sent_len, -1)
		word_list.append(char_features)

		word_embs = torch.cat(word_list, 2)
		word_represent = self.drop(word_embs)
		return word_represent
