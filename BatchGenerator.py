import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time
import os,sys
from collections import Counter
import punctuation
import datetime
import pyIO
import numpy as np
import pickle

# ** 3.build the data generator
class BatchGenerator(object):
    def __init__(self, X, y, shuffle=False, fill_word_id = -1):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        self.fill_word_id = fill_word_id
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def get_valid_pos_x(self, x_list):
        for i,v in enumerate(x_list):
            pos = len(x_list) - 1 - i
            if x_list[pos] != self.fill_word_id:
                return pos
        return 0

    def next_batch(self, batch_size):
        timestep_size = max_len = punctuation.get_timestep_size()

        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch

        offset = 0
        index_list = []
        start_pos = 0
        weight_change_list = []

        for i in range(len(self._X[start:end])):
            x = self._X[start:end][i]
            y = self._y[start:end][i]
            offset += sum([1 for e in x if e == 0])

            pos = self.get_valid_pos_x(x)
            tmp_list = [start_pos + e for e in range(pos+1)]
            index_list.extend(tmp_list)
            start_pos += len(x)

            tmp_list = [1.0 for e in range(len(x))]
            for k in range(len(tmp_list)):
                if y[k] > 0:
                    tmp_list[k] = 10.0
                else:
                    tmp_list[k] = 5.0

                if k > pos:
                    tmp_list[k] = 1.0

            weight_change_list.append(tmp_list)

        ###返回字符的数量，进行核对，避免出现错误统计
        batch_cnt_punc_dict = {}
        for i in range(len(punctuation.get_punc_list())):
            batch_cnt_punc_dict['%d'%i] = 0

        for i in range(len(self._y[start:end])):
            y = self._y[start:end][i]
            for v in y:
                if v >= 0:
                    v = int(v)
                    batch_cnt_punc_dict['%s'%v] += 1

        return self._X[start:end], self._y[start:end], offset, np.array(index_list).reshape(-1,1), np.array(weight_change_list).reshape(-1, timestep_size), batch_cnt_punc_dict
