# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-06-03
"""
import os
import pickle
from model.cnn_lstm_crf import CnnLstmConfig
from utils.functions import normalize_word, load_vocab
import numpy as np
from constants import ROOT_PATH

train_file = os.path.join(ROOT_PATH, 'dataset/char_level/train.txt')
dev_file = os.path.join(ROOT_PATH, 'dataset/char_level/dev.txt')
test_file = os.path.join(ROOT_PATH, 'dataset/char_level/test.txt')

# train_file = os.path.join(ROOT_PATH, 'dataset/char_level/debug/train.txt')
# dev_file = os.path.join(ROOT_PATH, 'dataset/char_level/debug/dev.txt')
# test_file = os.path.join(ROOT_PATH, 'dataset/char_level/debug/test.txt')

alphabet_file = os.path.join(ROOT_PATH, 'dataset/alphabet.dset')


class Data(object):
	def __init__(self):
		self.char_alphabet, self.word_alphabet, self.feat_alphabet, self.label_alphabet = {}, {}, {}, {}
		self.train_texts, self.dev_texts, self.test_texts = [], [], []
		self.train_ids, self.dev_ids, self.test_ids = [], [], []
		# self.batch_size = 10
		self.pretrain_feature_embeddings = None
		self.load_alphabet()
		self.read_instance(train_file)
		self.read_instance(dev_file)
		self.read_instance(test_file)
		self.feature_emb_dim = 5

		self.char_alphabet_size = len(self.char_alphabet) + 1
		self.word_alphabet_size = len(self.word_alphabet) + 1
		self.label_alphabet_size = len(self.label_alphabet) + 1
		self.feat_alphabet_size = len(self.feat_alphabet) + 1

	def load_alphabet(self):
		f = open(alphabet_file, 'rb')
		self.char_alphabet = pickle.load(f)
		self.word_alphabet = pickle.load(f)
		self.feat_alphabet = pickle.load(f)
		self.label_alphabet = pickle.load(f)
		self.pretrain_feature_embeddings = pickle.load(f)
		_ = pickle.load(f)
		self.feature_emb_dim = pickle.load(f)
		f.close()

	def read_instance(self, input_file):
		chars, labels = [], []
		char_ids, label_ids = [], []
		name = input_file.split('/')[-1].split('.')[0]
		with open(input_file, 'r') as rf:
			for line in rf:
				if len(line) > 1:
					pairs = line.strip().split()

					char = pairs[0]
					char = normalize_word(char)  # 如果word由数字构成，转为'0'
					label = pairs[-1]

					chars.append(char)
					labels.append(label)

					char_id = self.char_alphabet[char]
					label_id = self.label_alphabet[label]

					char_ids.append(char_id)
					label_ids.append(label_id)

				else:
					if name == 'train' and len(chars) > 0:
						self.train_texts.append([chars, labels])
						self.train_ids.append([char_ids, label_ids])
					elif name == 'dev' and len(chars) > 0:
						self.dev_texts.append([chars, labels])
						self.dev_ids.append([char_ids, label_ids])
					elif name == 'test' and len(chars) > 0:
						self.test_texts.append([chars, labels])
						self.test_ids.append([char_ids, label_ids])

					chars, labels = [], []
					char_ids, label_ids = [], []
			# 防止漏掉最后一行样本
			if len(chars) > 0:
				if name == 'train':
					self.train_texts.append([chars, labels])
					self.train_ids.append([char_ids, label_ids])
				elif name == 'dev':
					self.dev_texts.append([chars, labels])
					self.dev_ids.append([char_ids, label_ids])
				elif name == 'test':
					self.test_texts.append([chars, labels])
					self.test_ids.append([char_ids, label_ids])


if __name__ == '__main__':
	data = Data()
	pass
