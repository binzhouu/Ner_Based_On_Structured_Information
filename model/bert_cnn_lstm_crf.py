# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-09-25
"""

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from model.crf import CRF
import os
import numpy as np
from pytorch_pretrained_bert import BertModel

head_path, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))


class BertCLCConfig(object):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	gpu = True if device.type == 'cuda' else False

	batch_size = 10
	lr = 3e-3
	lr_decay = 0.05
	l2 = 1e-8
	momentum = 0
	epoch = 100
	dropout = 0.5
	max_sent_length = 250

	word_emb_dim = 300
	char_emb_dim = 768
	feature_num = 1
	hidden_dim = 200
	char_hidden_dim = 50
	feature_emb_dim = 5
	pretrain_char_embedding = None

	model_path = 'torch_model/bert_cnn_lstm_crf/best_model'
	bert_path = 'bert-base-chinese-pytorch'
	bert_wmm_path = 'chinese_wwm_pytorch'
	bert_vocab_file = 'chinese_wwm_pytorch/vocab.txt'


config = BertCLCConfig()


class BertCLC(nn.Module):
	def __init__(self, data):
		super(BertCLC, self).__init__()

		self.char_embeddings = BertModel.from_pretrained(config.bert_path)
		self.char_drop = nn.Dropout(config.dropout)
		self.char_cnn = nn.Conv1d(
			in_channels=config.char_emb_dim, out_channels=config.char_hidden_dim, kernel_size=3, padding=1)

		self.word_embeddings = nn.Embedding(data.word_alphabet_size, config.word_emb_dim)
		self.word_embeddings.weight.data.copy_(
			torch.from_numpy(self.random_embedding(data.word_alphabet_size, config.word_emb_dim)))
		self.word_drop = nn.Dropout(config.dropout)

		self.feature_embeddings = nn.Embedding(data.feat_alphabet_size, config.feature_emb_dim)

		if len(data.pretrain_feature_embeddings) > 1:
			self.feature_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_feature_embeddings))

		self.lstm = nn.LSTM(
			config.char_hidden_dim + config.word_emb_dim + config.feature_emb_dim, config.hidden_dim//2,
			num_layers=1, batch_first=True, bidirectional=True)
		self.droplstm = nn.Dropout(config.dropout)

		self.hidden2tag = nn.Linear(config.hidden_dim, data.label_alphabet_size + 2)  # label_size + 2 (crf的start和end)

		self.crf = CRF(data.label_alphabet_size, config.gpu)

	def forward(self, batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, batch_char_mask=None, batch_label=None):
		char_batch_size = batch_char.size(0)
		with torch.no_grad():
			char_embeds, _ = self.char_embeddings(batch_char, attention_mask=batch_char_mask, output_all_encoded_layers=False)
		char_embeds = self.char_drop(char_embeds)
		char_embeds = char_embeds.transpose(1, 2)  # 将max_length和embedding_dim转置
		char_cnn_out = self.char_cnn(char_embeds)  #
		char_cnn_out = torch.max_pool1d(char_cnn_out, kernel_size=char_cnn_out.size(2)).view(char_batch_size, -1)
		char_cnn_out = char_cnn_out[batch_charrecover]  # 还原排序之前的batch
		char_features = char_cnn_out.view(batch_word.size(0), batch_word.size(1), -1)

		feat_embs = self.feature_embeddings(batch_features)

		word_embs = self.word_embeddings(batch_word)
		word_embs = torch.cat([word_embs, char_features, feat_embs], 2)
		word_represent = self.word_drop(word_embs)

		packed_words = pack_padded_sequence(word_represent, batch_wordlen.cpu().numpy(), batch_first=True)
		hidden = None
		lstm_out, hidden = self.lstm(packed_words, hidden)
		lstm_out, _ = pad_packed_sequence(lstm_out)
		lstm_out = self.droplstm(lstm_out.transpose(1, 0))

		outputs = self.hidden2tag(lstm_out)

		if batch_label is not None:
			total_loss = self.crf.neg_log_likelihood_loss(outputs, mask, batch_label)
			scores, tag_seq = self.crf._viterbi_decode(outputs, mask)
			return total_loss, tag_seq
		else:
			scores, tag_seq = self.crf._viterbi_decode(outputs, mask)
			return tag_seq

	def random_embedding(self, vocab_size, embedding_dim):
		pretrain_emb = np.empty([vocab_size, embedding_dim])
		scale = np.sqrt(3.0 / embedding_dim)
		for index in range(vocab_size):
			pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
		return pretrain_emb
