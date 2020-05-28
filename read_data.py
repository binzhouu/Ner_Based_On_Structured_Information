# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-09-13
"""
import os
import pickle
from model.cnn_lstm_crf import CnnLstmConfig
from utils.functions import normalize_word, load_vocab
import numpy as np

config = CnnLstmConfig()

head_path, _ = os.path.split(os.path.abspath(__file__))

# train_file = os.path.join(head_path, 'dataset/train_lexi.txt')
# dev_file = os.path.join(head_path, 'dataset/dev_lexi.txt')
# test_file = os.path.join(head_path, 'dataset/test_lexi.txt')

train_file = os.path.join(head_path, 'dataset/debug/train_lexi.txt')
dev_file = os.path.join(head_path, 'dataset/debug/dev_lexi.txt')
test_file = os.path.join(head_path, 'dataset/debug/test_lexi.txt')

alphabet_file = os.path.join(head_path, 'dataset/alphabet.dset')

# bert_vocab_file = os.path.join(head_path, 'chinese_wwm_pytorch/vocab.txt')
# vocab = load_vocab(bert_vocab_file)


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
        chars, words, feats, labels = [], [], [], []
        char_ids, word_ids, feat_ids, label_ids = [], [], [], []
        name = input_file.split('/')[-1].split('_')[0]
        with open(input_file, 'r') as rf:
            for line in rf:
                if len(line) > 3:
                    pairs = line.strip().split()

                    word = pairs[0]
                    word = normalize_word(word)  # 如果word由数字构成，转为'0'
                    feat = pairs[1].split(']', 1)[-1]
                    label = pairs[-1]

                    words.append(word)
                    feats.append(feat)
                    labels.append(label)

                    word_id = self.word_alphabet[word]
                    feat_id = self.feat_alphabet[feat]
                    label_id = self.label_alphabet[label]

                    word_ids.append(word_id)
                    feat_ids.append(feat_id)
                    label_ids.append(label_id)

                    char_list = []
                    char_id = []
                    for char in word:
                        char_list.append(char)
                        char_id.append(self.char_alphabet[char])
                    chars.append(char_list)
                    char_ids.append(char_id)

                else:
                    if name == 'train' and len(words) > 0 and (len(words) < config.max_sent_length):
                        self.train_texts.append([chars, words, feats, labels])
                        self.train_ids.append([char_ids, word_ids, feat_ids, label_ids])
                    elif name == 'dev' and len(words) > 0 and (len(words) < config.max_sent_length):
                        self.dev_texts.append([chars, words, feats, labels])
                        self.dev_ids.append([char_ids, word_ids, feat_ids, label_ids])
                    elif name == 'test' and len(words) > 0 and (len(words) < config.max_sent_length):
                        self.test_texts.append([chars, words, feats, labels])
                        self.test_ids.append([char_ids, word_ids, feat_ids, label_ids])

                    chars, words, feats, labels = [], [], [], []
                    char_ids, word_ids, feat_ids, label_ids = [], [], [], []
            # 防止漏掉最后一行样本
            if len(words) > 0 and (len(words) < config.max_sent_length):
                if name == 'train':
                    self.train_texts.append([chars, words, feats, labels])
                    self.train_ids.append([char_ids, word_ids, feat_ids, label_ids])
                elif name == 'dev':
                    self.dev_texts.append([chars, words, feats, labels])
                    self.dev_ids.append([char_ids, word_ids, feat_ids, label_ids])
                elif name == 'test':
                    self.test_texts.append([chars, words, feats, labels])
                    self.test_ids.append([char_ids, word_ids, feat_ids, label_ids])


class BertData(object):
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
        _ = pickle.load(f)
        self.word_alphabet = pickle.load(f)
        self.feat_alphabet = pickle.load(f)
        self.label_alphabet = pickle.load(f)
        self.pretrain_feature_embeddings = pickle.load(f)
        _ = pickle.load(f)
        self.feature_emb_dim = pickle.load(f)
        f.close()

    def read_instance(self, input_file):
        chars, words, feats, labels = [], [], [], []
        char_ids, word_ids, feat_ids, label_ids = [], [], [], []
        name = input_file.split('/')[-1].split('_')[0]
        with open(input_file, 'r') as rf:
            for line in rf:
                if len(line) > 3:
                    pairs = line.strip().split()

                    word = pairs[0]
                    word = normalize_word(word)  # 如果word由数字构成，转为'0'
                    feat = pairs[1].split(']', 1)[-1]
                    label = pairs[-1]

                    words.append(word)
                    feats.append(feat)
                    labels.append(label)

                    word_id = self.word_alphabet[word]
                    feat_id = self.feat_alphabet[feat]
                    label_id = self.label_alphabet[label]

                    word_ids.append(word_id)
                    feat_ids.append(feat_id)
                    label_ids.append(label_id)

                    char_list = []
                    char_id = []
                    for char in word:
                        char_list.append(char)
                        char_id.append(vocab[char] if char in vocab else vocab['[UNK]'])
                    chars.append(char_list)
                    char_ids.append(char_id)

                else:
                    if name == 'train' and len(words) > 0 and (len(words) < config.max_sent_length):
                        self.train_texts.append([chars, words, feats, labels])
                        self.train_ids.append([char_ids, word_ids, feat_ids, label_ids])
                    elif name == 'dev' and len(words) > 0 and (len(words) < config.max_sent_length):
                        self.dev_texts.append([chars, words, feats, labels])
                        self.dev_ids.append([char_ids, word_ids, feat_ids, label_ids])
                    elif name == 'test' and len(words) > 0 and (len(words) < config.max_sent_length):
                        self.test_texts.append([chars, words, feats, labels])
                        self.test_ids.append([char_ids, word_ids, feat_ids, label_ids])

                    chars, words, feats, labels = [], [], [], []
                    char_ids, word_ids, feat_ids, label_ids = [], [], [], []

            # 防止漏掉最后一行样本
            if len(words) > 0 and (len(words) < config.max_sent_length):
                if name == 'train':
                    self.train_texts.append([chars, words, feats, labels])
                    self.train_ids.append([char_ids, word_ids, feat_ids, label_ids])
                elif name == 'dev':
                    self.dev_texts.append([chars, words, feats, labels])
                    self.dev_ids.append([char_ids, word_ids, feat_ids, label_ids])
                elif name == 'test':
                    self.test_texts.append([chars, words, feats, labels])
                    self.test_ids.append([char_ids, word_ids, feat_ids, label_ids])


if __name__ == '__main__':
    data = Data()
    # data = BertData()
    pass
