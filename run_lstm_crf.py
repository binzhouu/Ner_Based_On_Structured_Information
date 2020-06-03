# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-05-28
"""
import os
import logging
import logging.config
import yaml
import torch
import torch.optim as optim
import time
from constants import ROOT_PATH
from model.lstm_crf import BiLstmCrf
from read_data import Data
from utils.functions import batchify_sequence_labeling_with_label, predict_check, evalute_sequence_labeling

logger = logging.getLogger(__name__)
logging.config.fileConfig(os.path.join(ROOT_PATH, 'logging.conf'), disable_existing_loggers=False)

yml_path = os.path.join(ROOT_PATH, 'conf/model.yaml')


# 删除已存在的模型文件
def use_delete(if_del=True):
	def remove_exist_file(func):
		def wrapper(*args, **kwargs):
			if if_del:
				model_path = os.path.join(ROOT_PATH, 'saved_models/bilstm_crf')
				model_files = os.listdir(model_path)
				for i in model_files:
					if i.startswith('.'):
						continue
					name = os.path.join(model_path, i)
					os.remove(name)
			else:
				pass
			return func(*args, **kwargs)

		return wrapper

	return remove_exist_file


class Run(object):
	def __init__(self, configs):
		self.configs = configs
		self.encoder_type = 'bilstm_crf'

	@use_delete(True)
	def train(self, data):
		if self.encoder_type == 'bilstm_crf':
			model = BiLstmCrf(data, self.configs)
		else:
			logger.info('No model is selected!')
			return
		optimizer = optim.SGD(
			model.parameters(), lr=self.configs['lr'], momentum=self.configs['momentum'], weight_decay=self.configs['l2'])
		if self.configs['gpu']:
			model = model.cuda()
		best_dev = -10
		last_improved = 0

		for idx in range(self.configs['epoch']):
			epoch_start = time.time()
			temp_start = epoch_start
			logger.info("Epoch: %s/%s" % (idx, self.configs['epoch']))
			optimizer = self.lr_decay(optimizer, idx, self.configs['lr_decay'], self.configs['lr'])

			sample_loss = 0
			total_loss = 0
			right_token = 0
			whole_token = 0
			logger.info('first input word list: %s, %s' % (data.train_texts[0][1], data.train_ids[0][1]))

			model.train()
			model.zero_grad()
			batch_size = self.configs['batch_size']
			train_num = len(data.train_ids)
			logger.info('batch_size: %s, train_num: %s' % (batch_size, train_num))
			total_batch = train_num // batch_size + 1

			for batch_id in range(total_batch):
				start = batch_id * batch_size
				end = (batch_id + 1) * batch_size
				if end > train_num:
					end = train_num
				instance = data.train_ids[start:end]
				if not instance:
					continue
				batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, \
					batch_label, mask = batchify_sequence_labeling_with_label(instance, self.configs['gpu'], if_train=True)
				loss, tag_seq = model(batch_word, batch_wordlen, batch_wordrecover, mask, batch_label)
				right, whole = predict_check(tag_seq, batch_label, mask)
				right_token += right
				whole_token += whole
				sample_loss += loss.item()
				total_loss += loss.item()
				if end % 6400 == 0:
					temp_time = time.time()
					temp_cost = temp_time - temp_start
					temp_start = temp_time
					logger.info("Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
						end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
					if sample_loss > 1e8 or str(sample_loss) == "nan":
						print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
						exit(1)
					# 每次sample_loss要清零
					sample_loss = 0

				loss.backward()
				optimizer.step()
				model.zero_grad()
			temp_time = time.time()
			temp_cost = temp_time - temp_start
			logger.info("Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
				end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))

			epoch_finish = time.time()
			epoch_cost = epoch_finish - epoch_start
			logger.info("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
				idx, epoch_cost, train_num / epoch_cost, total_loss))

			# logger.info("Epoch: %s, Total loss: %s" % (idx, total_loss))
			speed, acc, p, r, f, _, _ = evalute_sequence_labeling(data, model, "dev", self.configs)
			dev_finish = time.time()
			dev_cost = dev_finish - epoch_finish

			current_score = f
			logger.info("Epoch: %s, Loss: %s, Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
				idx, total_loss, dev_cost, speed, acc, p, r, f))

			if current_score > best_dev:
				model_name = self.configs['model_path'] + '.' + str(idx) + '.model'
				torch.save(model.state_dict(), model_name)
				best_dev = current_score
				last_improved = idx

			speed, acc, p, r, f, _, _ = evalute_sequence_labeling(data, model, "test", self.configs)
			test_finish = time.time()
			test_cost = test_finish - dev_finish
			logger.info("Epoch: %s, Loss: %s, Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
					idx, total_loss, test_cost, speed, acc, p, r, f))

			# add early-stopping
			if idx - last_improved > self.configs['require_improvement']:
				logger.info('No optimization for %s epoch, auto-stopping' % self.configs['require_improvement'])
				break

	@staticmethod
	def lr_decay(optimzer, epoch, decay_rate, init_lr):
		lr = init_lr / (1 + decay_rate * epoch)
		logging.info("Learning rate is set as: %s", lr)
		for param_group in optimzer.param_groups:
			param_group['lr'] = lr
		return optimzer

	@classmethod
	def read_configs(cls):
		with open(yml_path, 'r') as rf:
			configs = yaml.load(rf, Loader=yaml.FullLoader)
		# 读取设备基本属性
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		gpu = True if device.type == 'cuda' else False
		map_location = 'cpu' if gpu is False else None
		configs.update({'device': device, 'gpu': gpu, 'map_location': map_location})
		# 读取model_num对应的模型超参数
		model_num = configs['model_num']
		for k, v in configs['model'][model_num].items():
			configs[k] = v
		del configs['model']

		return cls(configs)


if __name__ == '__main__':
	data = Data()
	run = Run.read_configs()
	run.train(data)
