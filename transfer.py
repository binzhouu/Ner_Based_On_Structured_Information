# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-09-03
"""
import time
import sys
import argparse
import random
import torch
import gc
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqlabel import SeqLabel
from utils.data import Data
import os
from utils.functions import read_instance

head_path, _ = os.path.split(os.path.abspath(__file__))

train_dir = head_path + '/dataset_with_feat/20190902_transfer/train_lexi_0902.txt'
dev_dir = head_path + '/dataset_with_feat/20190902_transfer/dev_lexi.txt'
test_dir = head_path + '/dataset_with_feat/20190902_transfer/test_lexi.txt'
raw_dir = head_path + '/dataset_with_feat/20190902_transfer/raw_lexi.txt'
config = head_path + '/config/transfer.config'
model_path = head_path + '/torch_model/lstmcrf.25.model'  # 原模型路径

data = Data()

data.load(head_path + '/torch_model/lstmcrf.dset')  # 读取原模型预处理，生成的data属性
data.read_config(config)  # 读取transfer.config，变更一些data属性
data.HP_gpu = torch.cuda.is_available()

train_texts, train_Ids = read_instance(
	train_dir, data.word_alphabet, data.char_alphabet, data.feature_alphabets, data.label_alphabet, data.number_normalized,
	data.MAX_SENTENCE_LENGTH, data.sentence_classification, data.split_token)

dev_texts, dev_Ids = read_instance(
	dev_dir, data.word_alphabet, data.char_alphabet, data.feature_alphabets, data.label_alphabet, data.number_normalized,
	data.MAX_SENTENCE_LENGTH, data.sentence_classification, data.split_token)

test_texts, test_Ids = read_instance(
	test_dir, data.word_alphabet, data.char_alphabet, data.feature_alphabets, data.label_alphabet, data.number_normalized,
	data.MAX_SENTENCE_LENGTH, data.sentence_classification, data.split_token)

raw_texts, raw_Ids = read_instance(raw_dir, data.word_alphabet, data.char_alphabet, data.feature_alphabets, data.label_alphabet, data.number_normalized,
	data.MAX_SENTENCE_LENGTH, data.sentence_classification, data.split_token)

data.train_texts, data.train_Ids, data.dev_texts, data.dev_Ids, data.test_texts, data.test_Ids = \
	train_texts, train_Ids, dev_texts, dev_Ids, test_texts, test_Ids

data.raw_texts, data.raw_Ids = raw_texts, raw_Ids

print('train_texts:', len(data.train_texts))
print('test_texts:', len(data.test_texts))
print('dev_texts:', len(data.dev_texts))
print('raw_texts:', len(data.raw_texts))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_location = 'cpu' if device.type == 'cpu' else None
print('device:', device)
print('map_location:', map_location)


def lr_decay(optimizer, epoch, decay_rate, init_lr):
	lr = init_lr / (1 + decay_rate * epoch)
	print(" Learning rate is set as:", lr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return optimizer


def batchify_with_label(input_batch_list, gpu, if_train=True, sentence_classification=False):
	if sentence_classification:
		return
	else:
		return batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train)


def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
	"""
		input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
			words: word ids for one sentence. (batch_size, sent_len)
			features: features ids for one sentence. (batch_size, sent_len, feature_num)
			chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
			labels: label ids for one sentence. (batch_size, sent_len)

		output:
			zero padding for word and char, with their batch length
			word_seq_tensor: (batch_size, max_sent_len) Variable
			feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
			word_seq_lengths: (batch_size,1) Tensor
			char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
			char_seq_lengths: (batch_size*max_sent_len,1) Tensor
			char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
			label_seq_tensor: (batch_size, max_sent_len)
			mask: (batch_size, max_sent_len)
	"""
	# print('batchify_sequence_labeling_with_label start...')
	# print('gpu:', gpu)
	batch_size = len(input_batch_list)
	words = [sent[0] for sent in input_batch_list]
	features = [np.asarray(sent[1]) for sent in input_batch_list]
	feature_num = len(features[0][0])
	chars = [sent[2] for sent in input_batch_list]
	labels = [sent[3] for sent in input_batch_list]
	word_seq_lengths = torch.LongTensor(list(map(len, words)))  # 每个句子的words的len
	max_seq_len = word_seq_lengths.max().item()  # 取words的len最打的句子长度作为max_seq_len
	word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
	label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
	# print('label_seq_tensor:', label_seq_tensor)
	feature_seq_tensors = []
	for idx in range(feature_num):  # 特征也是到word维度的，所以用word的max_seq_len
		feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long())
	mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
	for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
		seqlen = seqlen.item()
		word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
		label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
		# print('label_seq_tensor:', label_seq_tensor)
		mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
		for idy in range(feature_num):
			feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
	# 将样本按句子的长度(words的len排序)
	word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)  # word_perm_idx:排序之前的句子对应的index
	word_seq_tensor = word_seq_tensor[word_perm_idx]
	for idx in range(feature_num):
		feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

	label_seq_tensor = label_seq_tensor[word_perm_idx]
	mask = mask[word_perm_idx]
	# deal with char
	# pad_chars (batch_size, max_seq_len)
	pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]  # chars补0操作
	length_list = [list(map(len, pad_char)) for pad_char in pad_chars]  #
	max_word_len = max(map(max, length_list))
	char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
	char_seq_lengths = torch.LongTensor(length_list)
	# print('char_seq_lengths:', char_seq_lengths)
	for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
		for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
			# print len(word), wordlen
			# char_seq_tensor:(batch_size, max_seq_len, max_word_len)
			char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
	# char_seq_tensor:(batch_size * max_seqLen, max_word_len)
	char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
	# (10,40) -> (10 * 40)
	char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
	char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
	char_seq_tensor = char_seq_tensor[char_perm_idx]  # (400,14)
	_, char_seq_recover = char_perm_idx.sort(0, descending=False)
	_, word_seq_recover = word_perm_idx.sort(0, descending=False)
	if gpu:
		word_seq_tensor = word_seq_tensor.cuda()
		for idx in range(feature_num):
			feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
		word_seq_lengths = word_seq_lengths.cuda()
		word_seq_recover = word_seq_recover.cuda()
		label_seq_tensor = label_seq_tensor.cuda()
		char_seq_tensor = char_seq_tensor.cuda()
		char_seq_recover = char_seq_recover.cuda()
		mask = mask.cuda()
	return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, \
		char_seq_recover, label_seq_tensor, mask


def predict_check(pred_variable, gold_variable, mask_variable, sentence_classification=False):
	"""
		input:
			pred_variable (batch_size, sent_len): pred tag result, in numpy format
			gold_variable (batch_size, sent_len): gold result variable
			mask_variable (batch_size, sent_len): mask variable
	"""
	pred = pred_variable.cpu().data.numpy()
	gold = gold_variable.cpu().data.numpy()
	mask = mask_variable.cpu().data.numpy()
	overlaped = (pred == gold)
	if sentence_classification:
		# print(overlaped)
		# print(overlaped*pred)
		right_token = np.sum(overlaped)
		total_token = overlaped.shape[0]  # =batch_size
	else:
		right_token = np.sum(overlaped * mask)
		total_token = mask.sum()
	# print("right: %s, total: %s"%(right_token, total_token))
	return right_token, total_token


def evaluate(data, model, name, nbest=None):
	if name == "train":
		instances = data.train_Ids
	elif name == "dev":
		instances = data.dev_Ids
	elif name == 'test':
		instances = data.test_Ids
	elif name == 'raw':
		instances = data.raw_Ids
		# print(data.__dict__)
		# print('instances[1]:', instances[1])  # [word,features,char,label]
	else:
		print("Error: wrong evaluate name,", name)
		exit(1)
	# print('data.__dict__:', data.__dict__)
	right_token = 0
	whole_token = 0
	nbest_pred_results = []
	pred_scores = []
	pred_results = []
	gold_results = []
	# set torch_model in eval torch_model
	model.eval()
	batch_size = data.HP_batch_size  # 10
	# print(batch_size)
	start_time = time.time()
	train_num = len(instances)  # 112
	total_batch = train_num // batch_size + 1  # 把raw整体迭代完的batch数，不算epoch
	# print(total_batch)
	for batch_id in range(total_batch):  # 每10个instance为1个要预测的batch
		start = batch_id * batch_size
		end = (batch_id + 1) * batch_size
		if end > train_num:
			end = train_num  # 120>112,则end=112
		instance = instances[start:end]
		# 预测数据最终处理的格式如instance[0],raw.bmes的标签全部设为O
		# print('instance:', len(instance), instance[0])
		# for i in instance:
		#     print(len(i[0]), i[0])
		if not instance:
			continue
		# zero padding for word and char, 用batch中的max_seq_length
		# batchify_with_label:需要有实际的labels
		# 预测：
		batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, \
			batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
				instance, data.HP_gpu, False, data.sentence_classification)
		if nbest and not data.sentence_classification:
			# 预测结果的输入如下：
			scores, nbest_tag_seq = model.decode_nbest(
				batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest)
			# print('scores:', scores)
			# print('nbest_tag_seq:', nbest_tag_seq.shape, nbest_tag_seq)  # 每个sen,每个word的标签给出预测,nbest = shape[-1]
			# recover_nbest_label:将顺序调整与input对应，输出nbest个预测结果
			nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
			nbest_pred_results += nbest_pred_result
			pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()  # 调整pred_scores与input顺序一致
			# select the best sequence to evalurate
			tag_seq = nbest_tag_seq[:, :, 0]  # 只选了nbest的第一列
			# print('tag_seq:', tag_seq)  # 最终预测结果的序列
		else:
			tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
		# print("tag:",tag_seq)
		# recover_label:根据tag_seq还原出预测的label，根据batch_label还原出真实的label
		# batch_label：batch真实的label值
		pred_label, gold_label = recover_label(
			tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover, data.sentence_classification)
		# print('pred_label:', pred_label)  # 预测的标签
		# print('gold_label:', gold_label)  # 真实的标签
		pred_results += pred_label
		gold_results += gold_label
		# print('pred_results:', len(pred_results))
		# print('gold_results:', len(gold_results))
	print(name + ' ' + 'pred_results: ', len(pred_results))
	print(name + ' ' + 'gold_results:', len(gold_results))
	decode_time = time.time() - start_time
	# print('decode_time:', decode_time)
	speed = len(instances) / decode_time  # 每秒处理的句子数量
	acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
	if nbest and not data.sentence_classification:
		return speed, acc, p, r, f, nbest_pred_results, pred_scores
	return speed, acc, p, r, f, pred_results, pred_scores


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover,
				  sentence_classification=False):
	"""
		input:
			pred_variable (batch_size, sent_len): pred tag result
			gold_variable (batch_size, sent_len): gold result variable
			mask_variable (batch_size, sent_len): mask variable
	"""
	pred_variable = pred_variable[word_recover]
	gold_variable = gold_variable[word_recover]
	mask_variable = mask_variable[word_recover]
	batch_size = gold_variable.size(0)
	if sentence_classification:
		pred_tag = pred_variable.cpu().data.numpy().tolist()
		gold_tag = gold_variable.cpu().data.numpy().tolist()
		pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
		gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
	else:
		seq_len = gold_variable.size(1)
		mask = mask_variable.cpu().data.numpy()
		pred_tag = pred_variable.cpu().data.numpy()
		gold_tag = gold_variable.cpu().data.numpy()
		batch_size = mask.shape[0]
		pred_label = []
		gold_label = []
		for idx in range(batch_size):
			pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
			gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
			assert (len(pred) == len(gold))
			pred_label.append(pred)
			gold_label.append(gold)
	return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
	"""
		input:
			pred_variable (batch_size, sent_len, nbest): pred tag result
			mask_variable (batch_size, sent_len): mask variable
			word_recover (batch_size)
		output:
			nbest_pred_label list: [batch_size, nbest, each_seq_len]
	"""
	# exit(0)
	# print('word_recover:', word_recover)
	pred_variable = pred_variable[word_recover]  # 将预测结果的顺序调整与input一致了
	mask_variable = mask_variable[word_recover]
	batch_size = pred_variable.size(0)  # (10,41,nbest)
	seq_len = pred_variable.size(1)
	nbest = pred_variable.size(2)
	mask = mask_variable.cpu().data.numpy()  # numpy不能读取CUDA tensor，需要转化为CPU tensor
	# print('mask:', mask)
	pred_tag = pred_variable.cpu().data.numpy()
	# print(pred_tag)
	batch_size = mask.shape[0]
	pred_label = []
	for idx in range(batch_size):
		pred = []
		for idz in range(nbest):
			each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if
						 mask[idx][idy] != 0]
			# print('each_pred:', each_pred)
			pred.append(each_pred)
		# print(pred)  # nbest=n，就有n个pred
		pred_label.append(pred)
	# print(pred_label[0])
	return pred_label


def train(data):
	save_data_name = data.model_dir + ".dset"
	data.save(save_data_name)
	model = SeqLabel(data)
	# 加载预训练
	print('loading model %s' % model_path)
	model.load_state_dict(torch.load(model_path, map_location=map_location))
	print('data.seg:', data.seg)
	optimizer = ''
	if data.optimizer.lower() == "sgd":
		optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)

	best_dev = -10
	print('data.HP_gpu:', data.HP_gpu)
	for idx in range(data.HP_iteration):
		epoch_start = time.time()
		temp_start = epoch_start
		print("Epoch: %s/%s" % (idx, data.HP_iteration))
		if data.optimizer == "SGD":
			optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)

		instance_count = 0
		sample_id = 0
		sample_loss = 0  # 每500个batch清零
		total_loss = 0  # 一个epoch里的完整loss
		right_token = 0
		whole_token = 0
		# print("Before Shuffle: first input word list:", data.train_Ids[0][0])
		random.shuffle(data.train_Ids)
		print("Shuffle: first input word list:", data.train_Ids[0][0])
		model.train()
		model.zero_grad()
		batch_size = data.HP_batch_size
		# batch_id = 0
		train_num = len(data.train_Ids)
		print('train_num:', train_num)  # 训练样本的数量
		total_batch = train_num // batch_size + 1
		print('total_batch:', total_batch)

		for batch_id in range(total_batch):
			start = batch_id * batch_size
			end = (batch_id + 1) * batch_size
			if end > train_num:
				end = train_num
			instance = data.train_Ids[start:end]

			if not instance:
				continue

			batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, \
				batch_label, mask = batchify_with_label(instance, data.HP_gpu, True, data.sentence_classification)
			instance_count += 1
			loss, tag_seq = model.calculate_loss(
				batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label,
				mask)

			right, whole = predict_check(tag_seq, batch_label, mask, data.sentence_classification)  # pred与gold的校验
			right_token += right
			whole_token += whole
			sample_loss += loss.item()
			total_loss += loss.item()

			if end % 500 == 0:
				temp_time = time.time()
				temp_cost = temp_time - temp_start
				temp_start = temp_time
				print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
					end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
				if sample_loss > 1e8 or str(sample_loss) == "nan":
					print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
					exit(1)
				sys.stdout.flush()
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
		print("total_loss:", total_loss)
		if total_loss > 1e8 or str(total_loss) == "nan":
			print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
			exit(1)

		speed, acc, p, r, f, _, _ = evaluate(data, model, "dev")
		dev_finish = time.time()
		dev_cost = dev_finish - epoch_finish

		current_score = f
		print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))

		if current_score > best_dev:
			print("Exceed previous best f score:", best_dev)
			model_name = data.model_dir + '.' + str(idx) + ".model"
			print("Save current best torch_model in file:", model_name)  # 保存当前epoch结束的模型
			torch.save(model.state_dict(), model_name)
			best_dev = current_score
		# 每50轮保存一下
		if idx % 50 == 0:
			model_name = data.model_dir + '.' + str(idx) + ".model"
			print('Save every 50 epoch in file: %s' % model_name)
			torch.save(model.state_dict(), model_name)

		speed, acc, p, r, f, _, _ = evaluate(data, model, "test")
		test_finish = time.time()
		test_cost = test_finish - dev_finish
		print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_cost, speed, acc, p, r, f))

		# 对自己add对样本做一下evaluate:
		speed, acc, p, r, f, _, _ = evaluate(data, model, "raw")
		raw_finish = time.time()
		raw_cost = raw_finish - test_finish
		print("Raw: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (raw_cost, speed, acc, p, r, f))

		gc.collect()


if __name__ == '__main__':
	# train(data)
	model = SeqLabel(data)
	model.load_state_dict(torch.load('transfer_model/transfer.1.model', map_location=map_location))
	speed, acc, p, r, f, _, _ = evaluate(data, model, "raw")
	raw_finish = time.time()
