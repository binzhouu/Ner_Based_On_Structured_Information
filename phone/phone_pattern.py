# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-06-07
"""
import re
import os
import pickle


def write_pattern_pkl():
	head_path = os.path.dirname(__file__) + '/district_head.txt'
	phone_pattern_path = os.path.dirname(__file__) + '/phone_pattern.txt'
	pkl_path = os.path.dirname(__file__) + '/phone_pattern.pkl'
	dict_patterns = {}
	district_heads = ''
	phone_patterns = ''
	with open(head_path, 'r') as rf:
		for line in rf.readlines():
			district_heads += line.strip()
	with open(phone_pattern_path, 'r') as rf:
		for line in rf.readlines():
			phone_patterns += line
	dict_patterns['district_heads'] = district_heads
	dict_patterns['phone_patterns'] = phone_patterns
	# 将pattern写入pkl文件
	with open(pkl_path, 'wb') as wbf:
		pickle.dump(dict_patterns, wbf)


def load_phone_pattern():
	pkl_path = os.path.dirname(__file__) + '/phone_pattern.pkl'
	if not os.path.exists(pkl_path):
		write_pattern_pkl()
	with open(pkl_path, 'rb') as rbf:
		dict_patterns = pickle.load(rbf)
	district_head = dict_patterns['district_heads']
	# print(type(district_head))
	phone_patterns = dict_patterns['phone_patterns']
	# print(phone_patterns)
	re_phone_ch = '(?P<phone>' + '\n' + phone_patterns + '\n' + ')'
	re_phone_ch = re_phone_ch % {'var': district_head}
	re_pattern = re.compile(re_phone_ch, re.VERBOSE)  # re.VERBOSE:详细模式。这个模式下正则表达式可以是多行，忽略空白字符，并可以加入注释
	# print(re_pattern)
	return re_pattern


if __name__ == '__main__':
	pattern = load_phone_pattern()
	m = pattern.search('400-8004567')
	print(m)
