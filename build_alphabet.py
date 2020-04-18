# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-08-27
"""
import os
import pickle
from utils.functions import build_pretrained_emb, normalize_word

head_path, _ = os.path.split(os.path.abspath(__file__))

train_file = os.path.join(head_path, 'dataset/train_lexi.txt')
dev_file = os.path.join(head_path, 'dataset/dev_lexi.txt')
test_file = os.path.join(head_path, 'dataset/test_lexi.txt')

# train_file = os.path.join(head_path, 'dataset/renmindata_test/train_lexi_set.txt')
# dev_file = os.path.join(head_path, 'dataset/renmindata_test/dev_lexi_set.txt')
# test_file = os.path.join(head_path, 'dataset/renmindata_test/test_lexi_set.txt')

feat_embedding_path = os.path.join(head_path, 'embedding/feat_emb.txt')


class Alphabet(object):
	def __init__(self):
		self.word_alphabet = dict()
		self.feat_alphabet = dict()
		self.label_alphabet = dict()
		self.char_alphabet = dict()
		self.words, self.feats, self.labels, self.chars = [], [], [], []

		self.add_alphabet(train_file)
		self.add_alphabet(dev_file)
		self.add_alphabet(test_file)
		self.build_alphabet()
		# 预训练的feat_emb
		self.pretrain_feature_embeddings, self.feature_emb_dim = build_pretrained_emb(
			feat_embedding_path, self.feat_alphabet)
		self.batch_size = 10

		self.write_alphabet()

	def add_alphabet(self, input_file):
		with open(input_file, 'r') as rf:
			for i, line in enumerate(rf):
				if len(line) > 2:
					pairs = line.strip().split()
					word = pairs[0]
					word = normalize_word(word)  # 由数字组成的word，直接置为'0'
					feat = pairs[1].replace('[Lexi]', '')
					label = pairs[-1]
					self.words.append(word)
					self.feats.append(feat)
					self.labels.append(label)
					for char in word:
						self.chars.append(char)
				# if i % 100000 == 0:
				# 	print('read line: %s' % i)

	def build_alphabet(self):
		self.words = list(set(self.words))
		self.feats = list(set(self.feats))
		self.labels = list(set(self.labels))
		self.chars = list(set(self.chars))
		# 将手工特征先转为int
		self.feats = [int(i) for i in self.feats]
		# self.words.sort()
		self.feats.sort()
		# self.labels.sort()
		# self.chars.sort()
		# 加入unknow
		self.words = ['/unk'] + self.words
		self.feats = ['/unk'] + self.feats
		self.chars = ['/unk'] + self.chars
		# 从1开始算id(0可能会影响crf的运算)
		for i, w in enumerate(self.words):
			self.word_alphabet[w] = i + 1
		for i, f in enumerate(self.feats):
			if not isinstance(f, str):
				f = str(f)
			self.feat_alphabet[f] = i + 1
		for i, l in enumerate(self.labels):
			self.label_alphabet[l] = i + 1
		for i, c in enumerate(self.chars):
			self.char_alphabet[c] = i + 1

	def write_alphabet(self):
		f = open(head_path + '/dataset/alphabet.dset', 'wb')
		pickle.dump(self.char_alphabet, f)
		pickle.dump(self.word_alphabet, f)
		pickle.dump(self.feat_alphabet, f)
		pickle.dump(self.label_alphabet, f)
		pickle.dump(self.pretrain_feature_embeddings, f)
		pickle.dump(self.batch_size, f)
		pickle.dump(self.feature_emb_dim, f)
		f.close()


if __name__ == '__main__':
	alphabet = Alphabet()
