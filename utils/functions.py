# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-09-14
"""
import numpy as np
import torch
import time
from model.bert_cnn_lstm_crf import BertCLCConfig

config = BertCLCConfig()


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
    batch_size = len(input_batch_list)
    words = [sent[1] for sent in input_batch_list]
    features = [sent[2] for sent in input_batch_list]
    chars = [sent[0] for sent in input_batch_list]
    labels = [sent[-1] for sent in input_batch_list]

    word_seq_lengths = torch.tensor(list(map(len, words)), dtype=torch.long)  # 一个batch中，每个句子len组成的list
    max_seq_len = word_seq_lengths.max().item()  # 取一个batch中，最大长度
    # padding前准备
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()  # 取batch中最大长度，最为padding
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    feature_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()

    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
    # padding，先把tensor全部zeros，再根据len填充
    for idx, (seq, label, seq_len) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seq_len.item()
        word_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
        label_seq_tensor[idx, :seqlen] = torch.tensor(label, dtype=torch.long)
        mask[idx, :seqlen] = torch.tensor([1] * seqlen, dtype=torch.long)  # words中非padding的部分mask
        feature_seq_tensor[idx, :seqlen] = torch.tensor(features[idx], dtype=torch.long)

    # 将样本按句子的长度(words的len降序)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]  # 按降序重排batch中的句子
    feature_seq_tensor = feature_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    # chars补0到最大words的句子长度 （"len(chars[idx])"和词的长度是一样的)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]  # 每个word的char数量 统计
    max_word_len = max(map(max, length_list))  # 一个batch中，所有词的最大char长度
    # padding前准备：char_seq_tensor是padding后的形状
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.tensor(length_list, dtype=torch.long)  # batch中，每个句子，每个word的char数量
    # 开始char的padding
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.tensor(word, dtype=torch.long)
    char_seq_tensor = char_seq_tensor[word_perm_idx]  # 按词的长度先排序
    char_seq_tensor = char_seq_tensor.view(batch_size * max_seq_len, -1)  # 变换模型中char_inputs的形状
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )  # 同样，降序

    # 按word的字数长度(chars的len降序)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)  # 字级别再降序
    char_seq_tensor = char_seq_tensor[char_perm_idx]  # 字级别的输入再降序

    # 还原顺序的index：
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)  # char_seq_tensor还原的index
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)  # word_seq_tensor还原的index

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        feature_seq_tensor = feature_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()

    return word_seq_tensor, feature_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, \
           char_seq_recover, label_seq_tensor, mask


def predict_check(pred_variable, gold_variable, mask_variable):
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token


def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_ids
    elif name == "dev":
        instances = data.dev_ids
    elif name == 'test':
        instances = data.test_ids
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    # set model in eval model
    model.eval()
    batch_size = config.batch_size  # 10
    # print(batch_size)
    start_time = time.time()
    train_num = len(instances)  # 112
    total_batch = train_num // batch_size + 1  # 把raw整体迭代完的batch数，不算epoch

    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start: end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, \
        batch_label, mask = batchify_sequence_labeling_with_label(instance, config.gpu, False)

        # 对char加上'CLS'和'SEP'标记
        vocab = load_vocab(config.bert_vocab_file)
        batch_char_list = []
        batch_input_mask = []
        for n in range(len(batch_char)):
            input_mask = []
            token_ids = batch_char[n]
            token_ids = [vocab['[CLS]']] + list(np.array(token_ids.cpu().numpy()))
            if 0 in token_ids:
                token_ids[token_ids.index(0)] = vocab['[SEP]']
                token_ids.append(0)
            else:
                token_ids = token_ids + [vocab['[SEP]']]
            # 处理mask
            for i in token_ids:
                if i != 0:
                    input_mask.append(1)
                else:
                    input_mask.append(0)
            batch_char_list.append(token_ids)
            batch_input_mask.append(input_mask)
        batch_char = torch.tensor(batch_char_list, dtype=torch.long)
        batch_char_mask = torch.tensor(batch_input_mask)

        if config.gpu:
            batch_char = batch_char.cuda()
            batch_char_mask = batch_char_mask.cuda()

        # 模型预测
        tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask,
                        batch_char_mask)
        # tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        pred_label, gold_label = recover_label(
            tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results)
    return speed, acc, p, r, f, pred_results, pred_scores


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]

    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    index2label = invers_dict(label_alphabet)
    for idx in range(batch_size):
        pred = [index2label[str(pred_tag[idx][idy])] for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [index2label[str(gold_tag[idx][idy])] for idy in range(seq_len) if mask[idx][idy] != 0]
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def invers_dict(label_alphabet):
    index2alphabet = {}
    for k, v in label_alphabet.items():
        if not isinstance(v, str):
            v = str(v)
        index2alphabet[v] = k
    return index2alphabet


def get_ner_fmeasure(golden_lists, predict_lists):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        gold_matrix = get_ner_BIO(golden_list)
        pred_matrix = get_ner_BIO(predict_list)

        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner

    right_num = len(right_full)  # 交集的数量
    golden_num = len(golden_full)  # 真实标签的数量
    predict_num = len(predict_full)  # 预测标签的数量

    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num  # 交集/预测出的标签数量
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num  # 交集/真实的标签数量
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (right_tag + 0.0) / all_tag

    return accuracy, precision, recall, f_measure


def get_ner_BIO(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def build_pretrained_emb(embedding_path, word_alphabet):
    embedd_dict = dict()
    if embedding_path:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = len(word_alphabet) + 1
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.nan_to_num(np.empty([alphabet_size, embedd_dim]))
    pretrain_emb[0] = np.zeros([1, embedd_dim])  # 此行embedding不用

    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.items():
        if word in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, perfect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    return pretrain_emb, embedd_dim


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1  # 根据emb文件中向量的维度，重新定义embedd_dim
            elif embedd_dim + 1 != len(tokens):
                # ignore illegal embedding line
                continue
            # assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])  # np.empty生成的数组没有实际意义，使用时需要被替换赋值
            embedd[:] = tokens[1:]
            first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


# 如果token由数字组成，直接转为0
def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():  # 如果字符串为数字组成，则为True
            # print('char:', char)
            new_word += '0'
        # print('new_word:', new_word)
        else:
            new_word += char
    return new_word


# 将bert的vocab读入：
def load_vocab(vocab_file):
    vocab = dict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as rf:
        for line in rf:
            token = line.strip()
            vocab[token] = index
            index += 1
    return vocab
