# _*_ coding:utf-8 _*_

import os
import random
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import warnings
from utils.common import init_logger

warnings.filterwarnings("ignore")
logger = init_logger()

class ECGTools:
    def __init__(self):
        pass

    def split_train_dev_data(self, seed):
        random = range(1, 11)
        random = np.random.RandomState(seed).permutation(random)
        return random[:8], random[8:]       # 70%作为训练集，30%作为验证集。

    # 使用resnet对全部的训练数据抽取特征，作为训练lgb模型的数据
    def get_random_num(self, seed):
        random = range(1, 11)
        random = np.random.RandomState(seed).permutation(random)
        return random

    def metrics_scores(self, y_true, y_pred, y_score):
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)
        return precision, recall, f1, auc, acc


    def find_optimal_threshold(self, y_true, y_score):
        thresholds = np.linspace(0, 1, 100)
        f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
        return thresholds[np.argmax(f1s)]


    def metrics_f1(self, y_true, y_score, find_optimal):
        if find_optimal:
            thresholds = np.linspace(0, 1, 100)
        else:
            thresholds = [0.5]
        f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
        return np.max(f1s)


    def metrics_f1s(self, y_trues, y_scores, find_optimal=True):
        f1s = []
        for i in range(y_trues.shape[1]):
            f1 = self.metrics_f1(y_trues[:, i], y_scores[:, i], find_optimal)
            f1s.append(f1)
        return np.array(f1s)

    def seed_torch(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)    # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)            # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def get_device(self, args):
        return torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")


tools = ECGTools()

