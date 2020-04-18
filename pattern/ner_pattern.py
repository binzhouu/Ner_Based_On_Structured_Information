# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-08-28
"""
import os
import re

head_path, _ = os.path.split(os.path.abspath(__file__))

badcase_file = os.path.join(head_path, 'badcase.txt')
call_pattern_file = os.path.join(head_path, 'call_pattern.txt')
misident_file = os.path.join(head_path, 'misident_pattern.txt')


class NER_PATTERN(object):
	def __init__(self):
		self.dict_badcase = dict()
		self.call_pattern = ''
		self.call_name_list = list()
		self.misident_pattern = ''
		self.misdent_name_list = list()

		self.load_badcase()
		self.load_call_pattern()
		self.load_misident_pattern()

	def load_badcase(self):
		with open(badcase_file, 'r') as rf:
			for line in rf:
				text, label = line.strip().split('||')
				self.dict_badcase[text] = label

	def load_call_pattern(self):
		with open(call_pattern_file, 'r') as rf:
			for i, line in enumerate(rf):
				self.call_pattern += line.strip()
				self.call_name_list.append('name' + str(i))
		self.call_pattern = re.compile(self.call_pattern)

	def load_misident_pattern(self):
		with open(misident_file, 'r') as rf:
			for i, line in enumerate(rf):
				self.misident_pattern += line.strip()
				self.misdent_name_list.append('name' + str(i))
		self.misident_pattern = re.compile(self.misident_pattern)

