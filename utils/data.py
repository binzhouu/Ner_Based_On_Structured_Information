# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-08-27
"""
import os
head_path, _ = os.path.split(os.path.abspath(__file__))
embed_file = os.path.join(head_path, 'embedding/feat_emb.txt')


def build_pretrain_embedding(embed_file, feat_alphabet, embedd_dim):
	embedd_dict = dict()
