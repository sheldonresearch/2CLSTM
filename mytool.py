#!/usr/bin/env python
# encoding: utf-8


"""
@version: python2.7
@author: Xiangguo Sun
@contact: sunxiangguodut@qq.com
@site: http://blog.csdn.net/github_36326955
@software: PyCharm
@file: mytool
@time: 17-8-15 下午1:35
"""
import os
import sys
import numpy as np
from sklearn.metrics import precision_score,accuracy_score


GLOVE_DIR = "/home/sunxiangguo/PycharmProjects/glove.6B/"

def get_embeddings():
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]  # 单词的字符表示
            coefs = np.asarray(values[1:], dtype='float32')  # 单词的向量表示
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index



def load_data_labels(TEXT_DATA_DIR):
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath)
                        #f = open(fpath,mode='latin-1')
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                    f.close()
                    labels.append(label_id)
    return texts, labels, labels_index

def log_results(path, accuracy, micro_precision, weighted_precision, macro_precision):
    with open(path, "a") as f:
        f.write("accuracy  for classification model - " + str(accuracy))
        f.write("\n")
        f.write("micro Precision score for classification model - ")
        f.write(str(micro_precision))
        f.write("\n")
        f.write("weighted Precision score for classification model - ")
        f.write(str(weighted_precision))
        f.write("\n")
        f.write("macro Precision score for classification model - ")
        f.write(str(macro_precision))
        f.write("\n\n")


def get_result(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy  for classification model - ", accuracy)

    micro_precision = precision_score(y_test, y_pred, average='micro')
    print("micro Precision score for classification model - ", micro_precision)

    weighted_precision = precision_score(y_test, y_pred, average='weighted')
    print("weighted Precision score for classification model - ", weighted_precision)

    macro_precision = precision_score(y_test, y_pred, average='weighted')
    print("macro Precision score for classification model - ", macro_precision)

    return accuracy, micro_precision, weighted_precision, macro_precision