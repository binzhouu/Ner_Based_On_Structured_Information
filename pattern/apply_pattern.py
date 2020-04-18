# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-08-30
"""


def pattern_match(pattern, input_text):
	if pattern.search(input_text):
		return True
	else:
		return False


def call_pattern_apply(pattern, input_text, result, name_list):
	m = pattern.search(input_text)
	parse_list = [m.group(name) for name in name_list]
	entity_tmp = [i for i in parse_list if i is not None]
	entity = entity_tmp[0]
	while entity[-1] in ['吧', '啊', '呢', '啦', '呀']:
		entity = entity.replace(entity[-1], '')
	pattern_start = input_text.index(entity)
	pattern_end = pattern_start + len(entity)
	pattern_result = ['PERSON', entity, (pattern_start, pattern_end)]
	# 判断是否与模型结果重合
	drop = list()
	if len(result) > 0:
		for i in result:
			pred_start = i[2][0]
			pred_end = i[2][1]
			if (
				pred_start <= pattern_start <= pred_end or pred_start <= pattern_end <= pred_end
				or pattern_start <= pred_start <= pattern_end or pattern_start <= pred_end <= pattern_end
			):
				drop.append(i)
		for i in drop:
			result.remove(i)
	result.append(pattern_result)
	return result


def misident_pattern_apply(pattern, input_text, result, name_list):
	m = pattern.search(input_text)
	parse_list = [m.group(name) for name in name_list]
	entity_tmp = [i for i in parse_list if i is not None]
	misident_entity = entity_tmp[0]
	# 模型结果如果有misident,删除:
	drop = []
	if len(result) > 0:
		for i in result:
			if misident_entity in i[1]:
				drop.append(i)
		for i in drop:
			result.remove(i)
	return result
