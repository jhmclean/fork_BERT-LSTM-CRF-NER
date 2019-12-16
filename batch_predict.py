# encoding=utf-8

"""
@Author: LAI
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import codecs
import pickle
import os
from datetime import datetime

from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser
args = get_args_parser()

import jieba
import jieba.posseg
import jieba.analyse

model_dir = './output'
bert_dir = './chinese_L-12_H-768_A-12'

is_training=False
use_one_hot_embeddings=False
batch_size=4

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None


print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

# 加载label->id的词典
with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list) + 1


graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_mask")
    pos_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="pos_ids")

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))

    # jzhang: 注意，如果加载的模型用到了lstm，则一定要设置lstm_size与加载模型的lstm_size相等，不然会报错
    (total_loss, logits, trans, pred_ids, best_score, lstm_output) = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        labels=None, num_labels=num_labels, use_one_hot_embeddings=False, pos_ids=pos_ids_p, dropout_rate=1.0, lstm_size=args.lstm_size)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))


tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)


def predict_batch(input_txts):
    """
    do batch prediction.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
        sen_predict: 字典列表，每一条句子的预测结果被存储成一个dict
            key列表：
            sentence: 原始句子
            pred_tokens: 模型对该句子每一个token的label的预测
            pred_bilstm_score: BiLSTM层输出的得分情况，n_token*n_label
            crf_score: CRF层输出的路径得分
        trans_result_df: 模型得出的转移矩阵, n_label*n_label
    """
    def convert(line):
        feature = convert_single_example(line, label_list, args.max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids],(batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, args.max_seq_length))
        label_ids = np.reshape([feature.label_ids],(batch_size, args.max_seq_length))

        return input_ids, input_mask, segment_ids, label_ids

    global graph
    with graph.as_default():
        print(id2label)

        input_sents = [tokenizer.tokenize(line) for line in input_txts]
        input_features = [convert_single_example(sent, label_list, args.max_seq_length, tokenizer) for sent in input_sents]
        input_inputids = [feature.input_ids for feature in input_features]
        input_inputmask = [feature.input_mask for feature in input_features]

        pos_map = get_pos_map()
        input_posids = [convert_single_example_pos(sent, pos_map, args.max_seq_length) for sent in input_sents]

        if (len(input_txts) % batch_size) != 0:
            pad_num = batch_size - (len(input_txts) % batch_size)
            input_inputids.extend([[0 for _ in range(args.max_seq_length)] for _ in range(pad_num)])
            input_inputmask.extend([[0 for _ in range(args.max_seq_length)] for _ in range(pad_num)])
            input_posids.extend([[0 for _ in range(args.max_seq_length)] for _ in range(pad_num)])

        input_inputids = np.reshape(input_inputids,(-1, batch_size, args.max_seq_length))
        input_inputmask = np.reshape(input_inputmask,(-1, batch_size, args.max_seq_length))
        input_posids = np.reshape(input_posids,(-1, batch_size, args.max_seq_length))
        
        pred_labels = []
        best_scores = []
        logitss = []
        for i in range(input_inputids.shape[0]):
            feed_dict = {input_ids_p: input_inputids[i],
                         input_mask_p: input_inputmask[i],
                         pos_ids_p: input_posids[i]}
            # run session get current feed_dict result
            pred_ids_result, best_score_result, logits_result, lstm_output_result = sess.run([pred_ids, best_score, logits, lstm_output], feed_dict)

            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            pred_labels.extend(pred_label_result)
            best_scores.extend(best_score_result)
            logitss.extend(logits_result)

        # 获取转移矩阵
        trans_result = sess.run(trans)
        labels_add = ['padding'] + list(list(id2label.values()))
        trans_result_df = pd.DataFrame(trans_result, index=labels_add, columns=labels_add)
        sen_predict = []
        with open("NER_result.txt", "w") as write_result:
            for i in range(len(input_sents)):
                write_result.write("input sentence no.%d:\n" % i)
                write_result.write(str(input_sents[i]))
                write_result.write("\n")
                write_result.write("predicted tokens:\n")
                write_result.write(str(pred_labels[i]))
                write_result.write("\n")
                write_result.write("predicted BiLSTM score for each token:\n")
                bilstm_score = logitss[i][:len(pred_labels[i]), :]
                bilstm_score_df = pd.DataFrame(bilstm_score, columns=labels_add)
                write_result.write(str(bilstm_score_df.to_string()))
                write_result.write("\n")
                write_result.write("predicted CRF score:\n")
                write_result.write(str(best_scores[i]))
                write_result.write("\n\n")
                sen_info = {}
                sen_info['sentence'] = input_sents[i]
                sen_info['pred_tokens'] = pred_labels[i]
                sen_info['pred_bilstm_score'] = bilstm_score_df
                sen_info['crf_score'] = best_scores[i]
                sen_predict.append(sen_info)
            write_result.write("\n")
            write_result.write("Transition Matrix:\n")
            write_result.write(trans_result_df.to_string())
            write_result.write("\n\n")
            write_result.write("id2label:\n")
            write_result.write(str(id2label))

        # with open("NER_result.txt", "w") as write_result:
        #     for i in range(len(input_sents)):
        #         if len(input_sents[i]) != len(pred_labels[i]):
        #             raise ValueError("Length of sent is not the same as length of label")
        #
        #         for j in range(len(input_sents[i])):
        #             write_result.write(input_sents[i][j] + " " + pred_labels[i][j] + "\n")
        #         write_result.write("\n")

        return sen_predict, trans_result_df

def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result


def convert_single_example(example, label_list, max_seq_length, tokenizer):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(model_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature

def convert_single_example_pos(example, pos_map, max_seq_length):
    """
    Author:
        ZHANGJUNHAO304
    Args:
        feature:

    Returns:

    """
    sentence_seged = jieba.posseg.cut("".join(example))
    pos_seq = []
    # 标注方式：暂时采用比较简单的方式，例如，分词‘海域’的词性为n，则相应的token标记为['n', 'n']
    for x in sentence_seged:
        pos_seq.extend([x.flag] * len(x.word))

    # 序列截断
    if len(pos_seq) >= max_seq_length - 1:
        pos_seq = pos_seq[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志

    # 以[CLS]开头。暂时将[CLS]的词性设置为'x'
    pos_ids = [pos_map['x']]

    for p in pos_seq:
        try:
            pos_ids.append(pos_map[p])
        except:
            # 未知词归为'un'=unknown
            pos_ids.append(pos_map['un'])

    # 以[SEP]结尾。暂时将[SEP]的词性设置为'x'
    pos_ids.append(pos_map['x'])

    # padding
    padding_len = max_seq_length - len(pos_ids)
    pos_ids.extend([0] * padding_len)
    assert len(pos_ids) == max_seq_length
    return pos_ids

def get_pos_map():
    """
    词性列表参考1 https://www.cnblogs.com/adienhsuan/p/5674033.html
    词性列表参考2 https://gist.github.com/guodong000/54c74ed55575fa2305b6afd0cf46ba7c
    Author:
        ZHANGJUNHAO304
    Returns:

    """
    pos_list = [line.strip() for line in open("jieba_postags_complete.txt", "r")]
    # pos_list = ['Ag', 'a', 'ad', 'an', 'b', 'c', 'dg', 'd', 'e', 'f', 'g', 'h',
    #             'i', 'j', 'k', 'l', 'm', 'Ng', 'n', 'nr', 'ns', 'nt', 'o', 'p',
    #             'q', 'r', 's', 'tg', 't', 'u', 'uj', 'vg', 'v', 'vd', 'vn', 'w', 'x',
    #             'y', 'z', 'un']
    pos_map = {}
    for (i, pos) in enumerate(pos_list, 1):
        pos_map[pos] = i
    return pos_map


class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def merge(self):
        return self.__merge
    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types
    @word.setter
    def word(self, word):
        self.__word = word
    @start.setter
    def start(self, start):
        self.__start = start
    @end.setter
    def end(self, end):
        self.__end = end
    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)


class Result(object):
    def __init__(self, config):
        self.config = config
        self.person = []
        self.loc = []
        self.org = []
        self.others = []
    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        self.result_to_json(tokens, tags)
        return self.person, self.loc, self.org

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx+1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
        return item

    def append(self, word, start, end, tag):
        if tag == 'LOC':
            self.loc.append(Pair(word, start, end, 'LOC'))
        elif tag == 'PER':
            self.person.append(Pair(word, start, end, 'PER'))
        elif tag == 'ORG':
            self.org.append(Pair(word, start, end, 'ORG'))
        else:
            self.others.append(Pair(word, start, end, tag))


if __name__ == "__main__":
    input_txts = []
    with open("input.txt", "r", encoding='UTF-8') as read_txt:
        for line in read_txt:
            if line.strip() == "":
                continue
            input_txts.append(line.strip())
    sen_info, trans = predict_batch(input_txts)