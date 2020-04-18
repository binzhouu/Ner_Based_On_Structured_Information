# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-09-13
"""

from model.cnn_lstm_crf import CnnLstmConfig, CnnLstmCrf
from read_data import Data
import time
import random
import torch.optim as optim
from utils.functions import lr_decay, batchify_sequence_labeling_with_label, predict_check, evaluate
import torch
import logging
import sys
import numpy as np
from model.seqlabel import SeqLabel

logging.basicConfig(filemode='w')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('log/%s_log.txt' % sys.argv[0].split('/')[-1].replace('.py',''))
logger.addHandler(handler)

config = CnnLstmConfig()
map_location = 'cpu' if config.device.type == 'cpu' else None
gpu = False if config.device.type == 'cpu' else True
# 读取样本
data = Data()
data.gpu = gpu
data.char_alphabet_size = len(data.char_alphabet) + 1
data.word_alphabet_size = len(data.word_alphabet) + 1
data.label_alphabet_size = len(data.label_alphabet) + 1
data.feat_alphabet_size = len(data.feat_alphabet) + 1

data.dropout = 0.5
data.word_emb_dim = 300
data.char_emb_dim = 300
data.feature_num = 1
data.hidden_dim = 200
data.char_hidden_dim = 50
data.feature_emb_dim = 5
data.pretrain_char_embedding = None


seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def train():
	total_batch = 0
	# model = CnnLstmCrf(config)
	model = SeqLabel(data)
	optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.l2)
	if gpu:
		model = model.cuda()
	best_dev = -10

	for idx in range(config.epoch):
		epoch_start = time.time()
		temp_start = epoch_start
		print("Epoch: %s/%s" % (idx, config.epoch))
		optimizer = lr_decay(optimizer, idx, config.lr_decay, config.lr)
		instance_count = 0
		sample_id = 0
		sample_loss = 0  # 每500个batch清零
		total_loss = 0  # 一个epoch里的完整loss
		right_token = 0  # 一个epoch里预测正确的token数量
		whole_token = 0
		random.shuffle(data.train_ids)
		print("Shuffle: first input word list:", data.train_ids[0][1])

		model.train()
		model.zero_grad()
		batch_size = config.batch_size
		train_num = len(data.train_ids)
		print('batch_size:', batch_size, 'train_num:', train_num)
		total_batch = train_num // batch_size + 1

		for batch_id in range(total_batch):
			start = batch_id * batch_size
			end = (batch_id + 1) * batch_size
			if end > train_num:
				end = train_num
			instance = data.train_ids[start:end]  # [char,word,feat,label]
			if not instance:
				continue
			batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, \
				batch_label, mask = batchify_sequence_labeling_with_label(instance, gpu, if_train=True)
			# loss, tag_seq = model(batch_char, batch_word, batch_features, mask, batch_charrecover, batch_wordlen, batch_label)
			loss, tag_seq = model.calculate_loss(
				batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
			right, whole = predict_check(tag_seq, batch_label, mask)
			right_token += right
			whole_token += whole
			# print('right_token/whole_token:', right_token/whole_token)
			sample_loss += loss.item()
			total_loss += loss.item()
			if end % 6400 == 0:
				temp_time = time.time()
				temp_cost = temp_time - temp_start
				temp_start = temp_time
				print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
					end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
				if sample_loss > 1e8 or str(sample_loss) == "nan":
					print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
					exit(1)
				sample_loss = 0
			loss.backward()
			optimizer.step()
			model.zero_grad()
		temp_time = time.time()
		temp_cost = temp_time - temp_start
		print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
			end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))

		epoch_finish = time.time()
		epoch_cost = epoch_finish - epoch_start
		print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
			idx, epoch_cost, train_num / epoch_cost, total_loss))
		if total_loss > 1e8 or str(total_loss) == "nan":
			print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
			exit(1)
		logger.info("Epoch: %s, Total loss: %s" % (idx, total_loss))
		speed, acc, p, r, f, _, _ = evaluate(data, model, "dev")
		dev_finish = time.time()
		dev_cost = dev_finish - epoch_finish

		current_score = f
		print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))
		logger.info(
			"Epoch: %s, Loss: %s, Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
				idx, total_loss, dev_cost, speed, acc, p, r, f))
		if current_score > best_dev:
			model_name = config.model_path + '.' + str(idx) + '.model'
			torch.save(model.state_dict(), model_name)
			best_dev = current_score
			# logger.info("data:dev, epoch:%s, f1:%s, precision:%s, recall:%s" % (idx, current_score, p, r))

		speed, acc, p, r, f, _, _ = evaluate(data, model, "test")
		test_finish = time.time()
		test_cost = test_finish - dev_finish
		logger.info("Epoch: %s, Loss: %s, Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
			idx, total_loss, test_cost, speed, acc, p, r, f))


if __name__ == '__main__':
	train()
