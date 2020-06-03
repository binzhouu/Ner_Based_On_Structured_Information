# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-06-03
"""
import os
from constants import ROOT_PATH

train_source_file = os.path.join(ROOT_PATH, 'dataset/train_lexi.txt')
test_source_file = os.path.join(ROOT_PATH, 'dataset/test_lexi.txt')
dev_source_file = os.path.join(ROOT_PATH, 'dataset/dev_lexi.txt')

train_output_file = os.path.join(ROOT_PATH, 'dataset/char_level/train.txt')
test_output_file = os.path.join(ROOT_PATH, 'dataset/char_level/test.txt')
dev_output_file = os.path.join(ROOT_PATH, 'dataset/char_level/dev.txt')


# 将词级别的标签数据转为字符级别
# {'B-PERSON': 1, 'I-PERSON': 2, 'I-LOC': 3, 'I-SHO': 4, 'I-ORG': 5, 'O': 6, 'B-SHO': 7, 'B-LOC': 8, 'B-ORG': 9}
def write_char_level_data(resource_file, output_file):
	with open(resource_file, 'r') as rf:
		all_chars, all_char_labels = [], []
		chars, char_labels = [], []
		for line in rf:
			if len(line) > 3:
				pairs = line.strip().split()
				word, word_label = pairs[0], pairs[-1]
				char = list(word)
				char_label = ['O' for _ in char]
				if len(char_label) > 1:
					if word_label.startswith('I'):
						char_label = [word_label for _ in char_label]
					elif word_label == 'B-PERSON':
						char_label[0] = 'B-PERSON'
						char_label[1:] = ['I-PERSON' for _ in char_label[1:]]
					elif word_label == 'B-SHO':
						char_label[0] = 'B-SHO'
						char_label[1:] = ['I-SHO' for _ in char_label[1:]]
					elif word_label == 'B-ORG':
						char_label[0] = 'B-ORG'
						char_label[1:] = ['I-ORG' for _ in char_label[1:]]
					elif word_label == 'B-LOC':
						char_label[0] = 'B-LOC'
						char_label[1:] = ['I-LOC' for _ in char_label[1:]]
					else:
						pass
				else:
					char_label[0] = word_label
				chars += char
				char_labels += char_label
			else:
				all_chars.append(chars)
				all_char_labels.append(char_labels)
				chars, char_labels = [], []

	with open(output_file, 'w') as wf:
		for chars, char_labels in zip(all_chars, all_char_labels):
			for char, char_label in zip(chars, char_labels):
				wf.write(char + ' ' + char_label + '\n')
			wf.write('\n')


if __name__ == '__main__':
	write_char_level_data(train_source_file, train_output_file)
	write_char_level_data(dev_source_file, dev_output_file)
	write_char_level_data(test_source_file, test_output_file)
