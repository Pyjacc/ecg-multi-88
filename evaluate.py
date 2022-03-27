# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 14:39
# @Author  : Qinglong
# @File    : evaluate.py.py
# @Description: In User Settings Edit

'''
calculate real precision, recall, f1 scores
'''

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from utils.configures import args

def get_matrix():
    # 1. 读取预测的结果和真实的结果
    answer_path = args.answer_lgb_4_model       # todo: 根据想要测试的结果文件传入不同的文件路径

    true_labels = []
    predict_labels = []

    true_df = pd.read_csv(args.test_label_csv, sep="\t")
    for _, true_rows in true_df.iterrows():
        true_labels.append(true_rows[0])

    predict_df = pd.read_csv(answer_path, sep="\t")
    for _, predict_rows in predict_df.iterrows():
        predict_labels.append(predict_rows[0])

    assert len(true_labels) == len(predict_labels)


    # 2. 将label转换为(4000, 12)的矩阵
    length = len(true_labels)
    true_matrix = np.zeros((length, len(args.ecg_classes)))      # (4000, 12)
    pred_matrix = np.zeros((length, len(args.ecg_classes)))      # (4000, 12)

    for row in range(len(true_labels)):
        label = true_labels[row].split(",")
        for col in range(1, 13):
            true_matrix[row][col-1] = label[col]

    for row in range(len(predict_labels)):
        label = predict_labels[row].split(",")
        for col in range(1, 13):
            pred_matrix[row][col-1] = label[col]

    return true_matrix, pred_matrix


# 计算12分类中，每一个分类的precision，recall， f1值，然后计算12个分类的均值
def cal_12_diseases_avg_f1():
    true_matrix, pred_matrix = get_matrix()
    true_matrix, pred_matrix = true_matrix.transpose(), pred_matrix.transpose()

    p_list = []
    r_list = []
    f1_list = []

    # 计算每个类别的precision，recall，f1值，然后计算均值f1
    for i in range(len(args.ecg_classes)):
        y_true = true_matrix[i]
        y_pred = pred_matrix[i]

        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)

    print("f1 list:", f1_list)
    return np.mean(p_list), np.mean(r_list), np.mean(f1_list)


# 计算完全正确匹配率
def cal_absolute_correct():
    true_matrix, pred_matrix = get_matrix()

    err_num = 0
    for row in range(len(true_matrix)):
        for col in range(len(args.ecg_classes)):
            if true_matrix[row][col] != pred_matrix[row][col]:
                err_num += 1
                break

    correct_rate = (len(true_matrix) - err_num) / len(true_matrix)
    return correct_rate



if __name__ == "__main__":
    avg_p, avg_r, avg_f1 = cal_12_diseases_avg_f1()        # 类别平均分数
    print("12 leads average precision: %.5f, recall: %.5f, f1: %.5f" % (avg_p, avg_r, avg_f1))

    rate = cal_absolute_correct()      # 完全正确率
    print("12 leads absolute correct rate: %.5f" % rate)
    print("score: %.5f" % (avg_f1 * 0.8 + rate * 0.25))





