# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 17:14
# @Author  : Qinglong
# @File    : preprocess.py.py
# @Description: In User Settings Edit

import pandas as pd
import os
import math
import scipy.io as sio
import numpy as np
import random
from utils.tools import logger
from utils.configures import args

random.seed(args.seed)

bad_list1 = ['TEST00822', 'TEST04543', 'TEST13782', 'TEST15129', 'TEST15479', 'TEST15647', 'TEST17056', 'TEST19076',
             'TEST19618', 'TEST20030', 'TEST20351', 'TEST21130', 'TEST23246', 'TEST12050', 'TEST03218', 'TEST14672',
             'TEST18380', 'TEST18843', 'TEST19309', 'TEST22179', 'TEST00596', 'TEST05669', 'TEST08032', 'TEST10050',
             'TEST11119', 'TEST14034', 'TEST15012', 'TEST19928']

bad_list2 = ['TEST05161', 'TEST09473', 'TEST06896', 'TEST01165', 'TEST05309', 'TEST07597', 'TEST08082', 'TEST08560',
             'TEST02237', 'TEST03232', 'TEST06427', 'TEST06142', 'TEST06047', 'TEST06374', 'TEST04269', 'TEST09725',
             'TEST03728', 'TEST03165', 'TEST09917', 'TEST08729', 'TEST03090', 'TEST04127', 'TEST07739', 'TEST04938',
             'TEST07217', 'TEST09704', 'TEST05869', 'TEST02517', 'TEST04343', 'TEST04527', 'TEST04325', 'TEST05120',
             'TEST05021', 'TEST00163', 'TEST06895', 'TEST09694', 'TEST06967', 'TEST07779', 'TEST07296', 'TEST00872',
             'TEST07015', 'TEST04783', 'TEST05467', 'TEST08477', 'TEST04347', 'TEST06595', 'TEST02977', 'TEST03530',
             'TEST04721']
bad_list = bad_list1 + bad_list2


# 本地算力上运行，获取训练集、验证集和测试集的labels
def get_train_dev_test_label_local():
    logger.info("get train labels start...")
    # 获取训练样本中病种有冲突的样本
    conflict_id = get_conflict_samples()

    # 1. 读取用作测试集和验证集的patient_id
    new_reference_path = args.trainreference_new
    test_data_file = args.test_data_txt
    dev_data_file = args.dev_data_txt
    mini_sample_file = args.mini_sample_txt
    test_reference_list = []
    dev_reference_list = []
    mini_sample_list = []  # 样本数为1的样本对应的patient_id

    # 测试集数据patient_id
    with open(test_data_file, "r", encoding="utf-8") as fr1:
        lines = fr1.readlines()
        fr1.close()

        for line in lines:
            line = line.split()
            test_reference_list.append(line[0])

    # 验证集数据patient_id
    with open(dev_data_file, "r", encoding="utf-8") as fr2:
        lines = fr2.readlines()
        fr2.close()

        for line in lines:
            line = line.split()
            dev_reference_list.append(line[0])

    # 样本数量为1的样本的patient_id
    with open(mini_sample_file, "r", encoding="utf-8") as fr3:
        lines = fr3.readlines()
        fr3.close()

        for line in lines:
            line = line.split()
            mini_sample_list.append(line[0])

    # 2. 从生成的trainreference.csv文件中提取中用作训练集和验证集的labels.csv
    df_reference = pd.read_csv(new_reference_path, sep='\t', header=None)
    length_train = 0
    length_dev = 0
    length_test = 0
    result_train = []
    result_dev = []
    result_test = []

    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]

        if patient_id not in mini_sample_list:  # 排除样本数量为1的样本
            if patient_id in test_reference_list:  # 测试数据集
                labels_test = [0] * len(args.ecg_classes)
                for idx in info[1:]:
                    if idx != "":
                        labels_test[int(idx) - 1] = 1

                if patient_id not in conflict_id:
                    length_test += 1
                    result_test.append([patient_id] + labels_test)
            elif patient_id in dev_reference_list:  # 验证集数据
                labels_dev = [0] * len(args.ecg_classes)
                for idx in info[1:]:
                    if idx != "":
                        labels_dev[int(idx) - 1] = 1

                if patient_id not in conflict_id:
                    length_dev += 1
                    result_dev.append([patient_id] + labels_dev)
            else:  # 训练集
                labels_train = [0] * len(args.ecg_classes)
                for idx in info[1:]:
                    if idx != "":
                        labels_train[int(idx) - 1] = 1

                if patient_id not in conflict_id:
                    length_train += 1
                    result_train.append([patient_id] + labels_train)

    # 生成训练数据对应的标签并保存到文件
    random_num_train = np.zeros(length_train, dtype=np.int8)
    for i in range(10):
        start = int(length_train * i / 10)
        end = int(length_train * (i + 1) / 10)
        random_num_train[start:end] = i + 1

    df_train = pd.DataFrame(data=result_train, columns=["patient_id"] + args.ecg_classes)
    df_train['random'] = np.random.RandomState(args.seed).permutation(random_num_train)
    columns = df_train.columns
    df_train[columns].to_csv(args.train_label_csv, index=None)

    # 生成验证集数据对应的标签并保存到文件
    random_num_dev = np.zeros(length_dev, dtype=np.int8)
    for i in range(10):
        start = int(length_dev * i / 10)
        end = int(length_dev * (i + 1) / 10)
        random_num_dev[start:end] = i + 1

    df_dev = pd.DataFrame(data=result_dev, columns=["patient_id"] + args.ecg_classes)
    df_dev['random'] = np.random.RandomState(args.seed).permutation(random_num_dev)
    columns = df_dev.columns
    df_dev[columns].to_csv(args.dev_label_csv, index=None)

    # 生成测试数据对应的标签并保存到文件
    random_num_test = np.zeros(length_test, dtype=np.int8)
    for i in range(10):
        start = int(length_test * i / 10)
        end = int(length_test * (i + 1) / 10)
        random_num_test[start:end] = i + 1

    df_test = pd.DataFrame(data=result_test, columns=["patient_id"] + args.ecg_classes)
    df_test['random'] = np.random.RandomState(args.seed).permutation(random_num_test)
    columns = df_test.columns
    df_test[columns].to_csv(args.test_label_csv, index=None)
    logger.info("get train labels success!, platform: local")


# 本地算力上运行，获取训练集、验证集的labels（没有测试集）
def get_train_dev_label_local():
    logger.info("get train and dev labels start...")
    # 获取训练样本中病种有冲突的样本
    conflict_id = get_conflict_samples()

    # 1. 读取用作测试集和验证集的patient_id
    dev_data_file = args.dev_data_txt
    mini_sample_file = args.mini_sample_txt
    dev_reference_list = []
    mini_sample_list = []  # 样本数为1的样本对应的patient_id

    # 验证集数据patient_id
    with open(dev_data_file, "r", encoding="utf-8") as fr2:
        lines = fr2.readlines()
        fr2.close()

        for line in lines:
            line = line.split()
            dev_reference_list.append(line[0])

    # 样本数量为1的样本的patient_id
    with open(mini_sample_file, "r", encoding="utf-8") as fr3:
        lines = fr3.readlines()
        fr3.close()

        for line in lines:
            line = line.split()
            mini_sample_list.append(line[0])

    # 2. 从trainreference.csv文件中提取中用作训练集和验证集的labels.csv
    df_reference = pd.read_csv(args.trainreference_new, sep='\t', header=None)
    length_train = 0
    length_dev = 0
    result_train = []
    result_dev = []

    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]

        if patient_id not in mini_sample_list:  # 排除样本数量为1的样本
            if patient_id in dev_reference_list:  # 验证集数据
                labels_dev = [0] * len(args.ecg_classes)
                for idx in info[1:]:
                    if idx != "":
                        labels_dev[int(idx) - 1] = 1

                if patient_id not in conflict_id:
                    length_dev += 1
                    result_dev.append([patient_id] + labels_dev)
            else:  # 训练集
                labels_train = [0] * len(args.ecg_classes)
                for idx in info[1:]:
                    if idx != "":
                        labels_train[int(idx) - 1] = 1

                if patient_id not in conflict_id:
                    length_train += 1
                    result_train.append([patient_id] + labels_train)

    # 生成训练数据对应的标签并保存到文件
    random_num_train = np.zeros(length_train, dtype=np.int8)
    for i in range(10):
        start = int(length_train * i / 10)
        end = int(length_train * (i + 1) / 10)
        random_num_train[start:end] = i + 1

    df_train = pd.DataFrame(data=result_train, columns=["patient_id"] + args.ecg_classes)
    df_train['random'] = np.random.RandomState(args.seed).permutation(random_num_train)
    columns = df_train.columns
    df_train[columns].to_csv(args.train_label_csv, index=None)

    # 生成验证集数据对应的标签并保存到文件
    random_num_dev = np.zeros(length_dev, dtype=np.int8)
    for i in range(10):
        start = int(length_dev * i / 10)
        end = int(length_dev * (i + 1) / 10)
        random_num_dev[start:end] = i + 1

    df_dev = pd.DataFrame(data=result_dev, columns=["patient_id"] + args.ecg_classes)
    df_dev['random'] = np.random.RandomState(args.seed).permutation(random_num_dev)
    columns = df_dev.columns
    df_dev[columns].to_csv(args.dev_label_csv, index=None)


# 获取训练others模型的训练集、验证集和测试集的labels
def get_train_dev_test_label_others_local():
    logger.info("get train labels start...")
    # 获取训练样本中病种有冲突的样本
    conflict_id = get_conflict_samples()

    # 1. 读取用作测试集和验证集的patient_id
    test_data_file = args.test_data_other_txt
    dev_data_file = args.dev_data_other_txt
    mini_sample_file = args.mini_sample_other_txt
    test_reference_list = []
    dev_reference_list = []
    mini_sample_list = []  # 样本数为1的样本对应的patient_id

    # 测试集数据patient_id
    with open(test_data_file, "r", encoding="utf-8") as fr1:
        lines = fr1.readlines()
        fr1.close()

        for line in lines:
            line = line.split()
            test_reference_list.append(line[0])

    # 验证集数据patient_id
    with open(dev_data_file, "r", encoding="utf-8") as fr2:
        lines = fr2.readlines()
        fr2.close()

        for line in lines:
            line = line.split()
            dev_reference_list.append(line[0])

    # 样本数量为1的样本的patient_id
    with open(mini_sample_file, "r", encoding="utf-8") as fr3:
        lines = fr3.readlines()
        fr3.close()

        for line in lines:
            line = line.split()
            mini_sample_list.append(line[0])

    # 2. 从other_trainreference.csv文件中提取中用作训练集和验证集的labels.csv
    df_reference = pd.read_csv(args.trainreference_others, sep='\t', header=None)  # todo
    length_train = 0
    length_dev = 0
    length_test = 0
    result_train = []
    result_dev = []
    result_test = []

    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        if patient_id not in mini_sample_list:  # 排除样本数量为1的样本
            if patient_id in test_reference_list:  # 测试数据集
                labels_test = [0]
                if str(12) in disease:
                    labels_test[0] = 1
                if patient_id not in conflict_id:
                    length_test += 1
                    result_test.append([patient_id] + labels_test)
            elif patient_id in dev_reference_list:  # 验证集数据
                labels_dev = [0]
                if str(12) in disease:
                    labels_dev[0] = 1
                if patient_id not in conflict_id:
                    length_dev += 1
                    result_dev.append([patient_id] + labels_dev)
            else:  # 训练集
                labels_train = [0]
                if str(12) in disease:
                    labels_train[0] = 1
                if patient_id not in conflict_id:
                    length_train += 1
                    result_train.append([patient_id] + labels_train)

    # 生成训练数据对应的标签并保存到文件
    random_num_train = np.zeros(length_train, dtype=np.int8)
    for i in range(10):
        start = int(length_train * i / 10)
        end = int(length_train * (i + 1) / 10)
        random_num_train[start:end] = i + 1

    df_train = pd.DataFrame(data=result_train, columns=["patient_id"] + args.ecg_classes_others)
    df_train['random'] = np.random.RandomState(args.seed).permutation(random_num_train)
    columns = df_train.columns
    df_train[columns].to_csv(args.train_label_others_csv, index=None)

    # 生成验证集数据对应的标签并保存到文件
    random_num_dev = np.zeros(length_dev, dtype=np.int8)
    for i in range(10):
        start = int(length_dev * i / 10)
        end = int(length_dev * (i + 1) / 10)
        random_num_dev[start:end] = i + 1

    df_dev = pd.DataFrame(data=result_dev, columns=["patient_id"] + args.ecg_classes_others)
    df_dev['random'] = np.random.RandomState(args.seed).permutation(random_num_dev)
    columns = df_dev.columns
    df_dev[columns].to_csv(args.dev_label_others_csv, index=None)

    # 生成测试数据对应的标签并保存到文件
    random_num_test = np.zeros(length_test, dtype=np.int8)
    for i in range(10):
        start = int(length_test * i / 10)
        end = int(length_test * (i + 1) / 10)
        random_num_test[start:end] = i + 1

    df_test = pd.DataFrame(data=result_test, columns=["patient_id"] + args.ecg_classes_others)
    df_test['random'] = np.random.RandomState(args.seed).permutation(random_num_test)
    columns = df_test.columns
    df_test[columns].to_csv(args.test_label_others_csv, index=None)
    logger.info("get train labels success!, platform: local")


# 获取训练others模型的训练集、验证集的labels（没有测试集）
def get_train_dev_label_others_local():
    logger.info("get train labels start...")
    # 获取训练样本中病种有冲突的样本
    conflict_id = get_conflict_samples()

    # 1. 读取用作测试集和验证集的patient_id
    dev_data_file = args.dev_data_other_txt
    mini_sample_file = args.mini_sample_other_txt
    dev_reference_list = []
    mini_sample_list = []  # 样本数为1的样本对应的patient_id

    # 验证集数据patient_id
    with open(dev_data_file, "r", encoding="utf-8") as fr2:
        lines = fr2.readlines()
        fr2.close()

        for line in lines:
            line = line.split()
            dev_reference_list.append(line[0])

    # 样本数量为1的样本的patient_id
    with open(mini_sample_file, "r", encoding="utf-8") as fr3:
        lines = fr3.readlines()
        fr3.close()

        for line in lines:
            line = line.split()
            mini_sample_list.append(line[0])

    # 2. 从other_trainreference.csv文件中提取中用作训练集和验证集的labels.csv
    df_reference = pd.read_csv(args.trainreference_others, sep='\t', header=None)  # todo
    length_train = 0
    length_dev = 0
    result_train = []
    result_dev = []

    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        if patient_id not in mini_sample_list:  # 排除样本数量为1的样本
            if patient_id in dev_reference_list:  # 验证集数据
                labels_dev = [0]
                if str(12) in disease:
                    labels_dev[0] = 1
                if patient_id not in conflict_id:
                    length_dev += 1
                    result_dev.append([patient_id] + labels_dev)
            else:  # 训练集
                labels_train = [0]
                if str(12) in disease:
                    labels_train[0] = 1
                if patient_id not in conflict_id:
                    length_train += 1
                    result_train.append([patient_id] + labels_train)

    # 生成训练数据对应的标签并保存到文件
    random_num_train = np.zeros(length_train, dtype=np.int8)
    for i in range(10):
        start = int(length_train * i / 10)
        end = int(length_train * (i + 1) / 10)
        random_num_train[start:end] = i + 1

    df_train = pd.DataFrame(data=result_train, columns=["patient_id"] + args.ecg_classes_others)
    df_train['random'] = np.random.RandomState(args.seed).permutation(random_num_train)
    columns = df_train.columns
    df_train[columns].to_csv(args.train_label_others_csv, index=None)

    # 生成验证集数据对应的标签并保存到文件
    random_num_dev = np.zeros(length_dev, dtype=np.int8)
    for i in range(10):
        start = int(length_dev * i / 10)
        end = int(length_dev * (i + 1) / 10)
        random_num_dev[start:end] = i + 1

    df_dev = pd.DataFrame(data=result_dev, columns=["patient_id"] + args.ecg_classes_others)
    df_dev['random'] = np.random.RandomState(args.seed).permutation(random_num_dev)
    columns = df_dev.columns
    df_dev[columns].to_csv(args.dev_label_others_csv, index=None)


# 获取训练normal模型的训练集、验证集和测试集的labels
def get_train_dev_test_label_normal_local():
    logger.info("get train labels start...")
    # 获取训练样本中病种有冲突的样本
    conflict_id = get_conflict_samples()

    # 1. 读取用作测试集和验证集的patient_id
    test_data_file = args.test_data_normal_txt
    dev_data_file = args.dev_data_normal_txt
    mini_sample_file = args.mini_sample_normal_txt
    test_reference_list = []
    dev_reference_list = []
    mini_sample_list = []  # 样本数为1的样本对应的patient_id

    # 测试集数据patient_id
    with open(test_data_file, "r", encoding="utf-8") as fr1:
        lines = fr1.readlines()
        fr1.close()

        for line in lines:
            line = line.split()
            test_reference_list.append(line[0])

    # 验证集数据patient_id
    with open(dev_data_file, "r", encoding="utf-8") as fr2:
        lines = fr2.readlines()
        fr2.close()

        for line in lines:
            line = line.split()
            dev_reference_list.append(line[0])

    # 样本数量为1的样本的patient_id
    with open(mini_sample_file, "r", encoding="utf-8") as fr3:
        lines = fr3.readlines()
        fr3.close()

        for line in lines:
            line = line.split()
            mini_sample_list.append(line[0])

    # 2. 从other_trainreference.csv文件中提取中用作训练集和验证集的labels.csv
    df_reference = pd.read_csv(args.trainreference_normal, sep='\t', header=None)  # todo
    length_train = 0
    length_dev = 0
    length_test = 0
    result_train = []
    result_dev = []
    result_test = []

    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        if patient_id not in mini_sample_list:  # 排除样本数量为1的样本
            if patient_id in test_reference_list:  # 测试数据集
                labels_test = [0]
                if str(1) in disease:
                    labels_test[0] = 1
                if patient_id not in conflict_id:
                    length_test += 1
                    result_test.append([patient_id] + labels_test)
            elif patient_id in dev_reference_list:  # 验证集数据
                labels_dev = [0]
                if str(1) in disease:
                    labels_dev[0] = 1
                if patient_id not in conflict_id:
                    length_dev += 1
                    result_dev.append([patient_id] + labels_dev)
            else:  # 训练集
                labels_train = [0]
                if str(1) in disease:
                    labels_train[0] = 1
                if patient_id not in conflict_id:
                    length_train += 1
                    result_train.append([patient_id] + labels_train)

    # 生成训练数据对应的标签并保存到文件
    random_num_train = np.zeros(length_train, dtype=np.int8)
    for i in range(10):
        start = int(length_train * i / 10)
        end = int(length_train * (i + 1) / 10)
        random_num_train[start:end] = i + 1

    df_train = pd.DataFrame(data=result_train, columns=["patient_id"] + args.ecg_classes_normal)
    df_train['random'] = np.random.RandomState(args.seed).permutation(random_num_train)
    columns = df_train.columns
    df_train[columns].to_csv(args.train_label_normal_csv, index=None)

    # 生成验证集数据对应的标签并保存到文件
    random_num_dev = np.zeros(length_dev, dtype=np.int8)
    for i in range(10):
        start = int(length_dev * i / 10)
        end = int(length_dev * (i + 1) / 10)
        random_num_dev[start:end] = i + 1

    df_dev = pd.DataFrame(data=result_dev, columns=["patient_id"] + args.ecg_classes_normal)
    df_dev['random'] = np.random.RandomState(args.seed).permutation(random_num_dev)
    columns = df_dev.columns
    df_dev[columns].to_csv(args.dev_label_normal_csv, index=None)

    # 生成测试数据对应的标签并保存到文件
    random_num_test = np.zeros(length_test, dtype=np.int8)
    for i in range(10):
        start = int(length_test * i / 10)
        end = int(length_test * (i + 1) / 10)
        random_num_test[start:end] = i + 1

    df_test = pd.DataFrame(data=result_test, columns=["patient_id"] + args.ecg_classes_normal)
    df_test['random'] = np.random.RandomState(args.seed).permutation(random_num_test)
    columns = df_test.columns
    df_test[columns].to_csv(args.test_label_normal_csv, index=None)
    logger.info("get train labels success!, platform: local")


# 获取训练normal模型的训练集、验证集的labels（没有测试集）
def get_train_dev_label_normal_local():
    logger.info("get train labels start...")
    # 获取训练样本中病种有冲突的样本
    conflict_id = get_conflict_samples()

    # 1. 读取用作测试集和验证集的patient_id
    dev_data_file = args.dev_data_normal_txt
    mini_sample_file = args.mini_sample_normal_txt
    dev_reference_list = []
    mini_sample_list = []  # 样本数为1的样本对应的patient_id

    # 验证集数据patient_id
    with open(dev_data_file, "r", encoding="utf-8") as fr2:
        lines = fr2.readlines()
        fr2.close()

        for line in lines:
            line = line.split()
            dev_reference_list.append(line[0])

    # 样本数量为1的样本的patient_id
    with open(mini_sample_file, "r", encoding="utf-8") as fr3:
        lines = fr3.readlines()
        fr3.close()

        for line in lines:
            line = line.split()
            mini_sample_list.append(line[0])

    # 2. 从other_trainreference.csv文件中提取中用作训练集和验证集的labels.csv
    df_reference = pd.read_csv(args.trainreference_normal, sep='\t', header=None)  # todo
    length_train = 0
    length_dev = 0
    result_train = []
    result_dev = []

    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        if patient_id not in mini_sample_list:  # 排除样本数量为1的样本
            if patient_id in dev_reference_list:  # 验证集数据
                labels_dev = [0]
                if str(1) in disease:
                    labels_dev[0] = 1
                if patient_id not in conflict_id:
                    length_dev += 1
                    result_dev.append([patient_id] + labels_dev)
            else:  # 训练集
                labels_train = [0]
                if str(1) in disease:
                    labels_train[0] = 1
                if patient_id not in conflict_id:
                    length_train += 1
                    result_train.append([patient_id] + labels_train)

    # 生成训练数据对应的标签并保存到文件
    random_num_train = np.zeros(length_train, dtype=np.int8)
    for i in range(10):
        start = int(length_train * i / 10)
        end = int(length_train * (i + 1) / 10)
        random_num_train[start:end] = i + 1

    df_train = pd.DataFrame(data=result_train, columns=["patient_id"] + args.ecg_classes_normal)
    df_train['random'] = np.random.RandomState(args.seed).permutation(random_num_train)
    columns = df_train.columns
    df_train[columns].to_csv(args.train_label_normal_csv, index=None)

    # 生成验证集数据对应的标签并保存到文件
    random_num_dev = np.zeros(length_dev, dtype=np.int8)
    for i in range(10):
        start = int(length_dev * i / 10)
        end = int(length_dev * (i + 1) / 10)
        random_num_dev[start:end] = i + 1

    df_dev = pd.DataFrame(data=result_dev, columns=["patient_id"] + args.ecg_classes_normal)
    df_dev['random'] = np.random.RandomState(args.seed).permutation(random_num_dev)
    columns = df_dev.columns
    df_dev[columns].to_csv(args.dev_label_normal_csv, index=None)


# aiwin训练平台上测试运行,将trainreference解析为train_labels
def get_train_label_aiwin():
    logger.info("get train labels start...")
    # 获取训练样本中病种有冲突的样本
    conflict_id = get_conflict_samples()

    result = []
    df_reference = pd.read_csv(args.trainreference, sep='\t', header=None)
    length = 0

    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]

        labels = [0] * len(args.ecg_classes)
        for idx in info[1:]:
            if idx != "":
                labels[int(idx) - 1] = 1

        if patient_id not in conflict_id:
            result.append([patient_id] + labels)
            length += 1

    random_num = np.zeros(length, dtype=np.int8)
    for i in range(10):
        start = int(length * i / 10)
        end = int(length * (i + 1) / 10)
        random_num[start:end] = i + 1

    df = pd.DataFrame(data=result, columns=["patient_id"] + args.ecg_classes)
    df['random'] = np.random.RandomState(args.seed).permutation(random_num)
    columns = df.columns
    df[columns].to_csv(args.train_label_csv, index=None)
    logger.info("get train labels success!, platform: aiwin")


# 从原始数据中获取出现冲突病种的样本id，并保存到txt中
def get_conflict_samples():
    logger.info("get conflict samples start...")

    patient_id_list = []
    df_reference = pd.read_csv(args.trainreference, sep='\t', header=None)

    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        # 获取出现了1号且同时出现了其他编号病的样本的id
        if str(1) in disease and len(disease) > 1:
            patient_id_list.append(patient_id)

        # 获取出同时出现了窦性心动过缓、窦性心动过速的样本的id
        if str(2) in disease and str(3) in disease:
            patient_id_list.append(patient_id)

        # 获取出现了窦性心动过缓、窦性心动过速、窦性心律不齐中的一种，且出现了 心房颤动 的样本的patient_id
        if str(2) in disease and str(5) in disease:
            patient_id_list.append(patient_id)
        elif "3" in disease and "5" in disease:
            patient_id_list.append(patient_id)
        elif "4" in disease and "5" in disease:
            patient_id_list.append(patient_id)

    logger.info("get conflict samples success!")
    return patient_id_list


# 获取两种病伴随出现的概率矩阵
def get_disease_matrix_prob():
    logger.info("get disease matrix start...")

    conflict_id = get_conflict_samples()
    disease_matrix = np.zeros((len(args.ecg_classes), len(args.ecg_classes)))
    disease_dict = dict()
    df_reference = pd.read_csv(args.trainreference_new, sep='\t', header=None)

    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        if patient_id not in conflict_id:
            for i in disease:
                for j in disease:
                    key = str(i) + "_" + str(j)
                    if key not in disease_dict.keys():
                        disease_dict[key] = 1
                    else:
                        disease_dict[key] += 1

    total_num = 0
    for key in disease_dict.keys():
        total_num += disease_dict[key]

    for key in disease_dict.keys():
        row = int(key.split("_")[0]) - 1
        col = int(key.split("_")[1]) - 1
        disease_matrix[row][col] = disease_dict[key] / total_num

    disease_matrix = disease_matrix.flatten().tolist()
    # df = pd.DataFrame()
    # df['probability'] = disease_matrix
    # df.to_csv(args.disease_prob_csv, index=None)

    logger.info("get conflict samples success!")
    return disease_matrix


# 从所有样本中采样训练集、验证集、测试集
def split_train_dev_test_data():
    # 第一步：采样测试集test data
    # 第二步：采样验证集dev data， 剩下的数据作为训练集train data
    # 第三步：从dev data和train data的总体中采样，用于寻找lgb的阈值
    # 第四步：并对小样本数据进行重采样，生成部分样本

    logger.info("split train test data start...")
    org_reference_path = args.trainreference
    new_reference_path = args.trainreference_new
    generate_sample_dir = args.train_data_dir  # todo: 保存生成样本的路径

    conflict_id = get_conflict_samples()
    conflict_id += bad_list  # 将病种有冲突的和图形效果很差的样本合并到一起，从数据集中去除
    disease_dict = dict()  # 保存所有的数据，key：病种编号组合出的字符串，value:patient id构成的list
    test_reat = 0.1  # 测试集占整个数据集的比例
    test_sample_thresholds = 10  # 采样阈值：从样本数大于10个的样本中采样测试集
    test_data = list()  # 保存测试集样本的patient_id
    train_dev_data_dict = dict()  # 保存去除测试集后的数据，作为训练集和验证集
    dev_reat = 0.2  # 验证集占训练集和验证集总数的比例
    dev_sample_thresholds = 5  # 采样阈值：从样本数大于5个的样本中采样验证集
    dev_data = list()  # 保存验证集样本的patient_id
    train_data_dict = dict()  # 保存去除验证集后的数据，作为训练集
    lgb_thresholds = 5  # 采样阈值：从样本数大于5个的样本中采样,用于寻找lgb的最佳阈值
    lgb_rate = 0.1  # 用于寻找lgb最佳阈值的样本占整个样本的比例
    lgb_threshold_data = list()  # 保存寻找lgb的最佳阈值的样本
    mini_sample_threshold = 1  # 小样本数阈值
    mini_sample_patient = []  # 统计病种组合对应的样本数为1的样本的patient_id
    mini_sample_key = []  # 统计病种组合对应的样本数为1的病种
    sample_threshold = 20  # 采样生成样本的阈值：经测试，样本增强到20效果最佳
    generate_num = 30000  # 生成样本的起始patient_id

    # 统计原始数据中病种的组合种类，及每个种类的数量
    df_reference = pd.read_csv(org_reference_path, sep='\t', header=None)
    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        # 将样本的病种合并为字符串，作为字典的key
        if patient_id not in conflict_id:
            key = ""
            for dis in disease:
                if dis != disease[-1]:
                    key += dis + "_"
                else:
                    key += dis

        # 将样本的patient_id构成的list作为字典的value
        if key not in disease_dict.keys():
            value = list()
            value.append(patient_id)
            disease_dict[key] = value
        else:
            disease_dict[key].append(patient_id)

    # 将病种组合写入文件，用于对预测的结果进行校正
    with open(args.disease_groups, "w", encoding='utf-8') as fw1:
        for key in disease_dict.keys():
            fw1.write(key)
            fw1.write('\n')
        fw1.close()

    # 统计样本数为1的样本对应的patient_id
    for key in disease_dict.keys():
        if len(disease_dict[key]) <= mini_sample_threshold:  # todo
            mini_sample_patient += disease_dict[key]
            mini_sample_key.append(key)

    with open(args.mini_sample_txt, "w", encoding="utf-8") as fw2:
        for sample in mini_sample_patient:
            fw2.write(sample)
            fw2.write("\n")
        fw2.close()

    # 第一步：采样出测试集，剩下的作为训练集和验证集
    for key in disease_dict.keys():
        if len(disease_dict[key]) >= test_sample_thresholds:
            test_len = math.floor(test_reat * len(disease_dict[key])) + 1  # 保证至少采样一个样本作为测试集
            random.shuffle(disease_dict[key])
            test_sample = disease_dict[key][:test_len]  # 获取测试集数据
            test_data += test_sample
            train_dev_sample = disease_dict[key][test_len:]  # 剩余部分作为训练集和验证集
            train_dev_data_dict[key] = train_dev_sample
        else:
            train_dev_data_dict[key] = disease_dict[key]

    # 第二步：采样出验证集，剩下的作为训练集
    for key in train_dev_data_dict.keys():
        if len(train_dev_data_dict[key]) >= dev_sample_thresholds:
            dev_len = math.floor(dev_reat * len(train_dev_data_dict[key])) + 1  # 保证至少采样一个样本作为验证集
            random.shuffle(train_dev_data_dict[key])
            dev_sample = train_dev_data_dict[key][:dev_len]  # 获取测试集数据
            dev_data += dev_sample
            train_sample = train_dev_data_dict[key][dev_len:]  # 剩余部分作为训练集
            train_data_dict[key] = train_sample
        else:
            train_data_dict[key] = train_dev_data_dict[key]

    # 第三步：从训练集和验证集的总体中采样，用于寻找lgb的阈值
    for key in train_dev_data_dict.keys():
        if len(train_dev_data_dict[key]) >= lgb_thresholds:
            lgb_len = math.floor(lgb_rate * len(train_dev_data_dict[key])) + 1  # 保证至少采样一个样本作为测试集
            random.shuffle(train_dev_data_dict[key])
            lgb_sample = train_dev_data_dict[key][:lgb_len]  # 获取测试集数据
            lgb_threshold_data += lgb_sample

    # 第四步：数据增强，生成小样本数据
    # 4.1 对训练样本进行处理，如果train_data_dict中每种组合病种的样本数量少于20，则从该类组合病种的样本中采样，生成20个样本
    generate_sample_dict = dict()  # key:生成的样本的patient_id，value：病种的组合

    # for key in train_data_dict.keys():
    #     if key not in mini_sample_key:      # 不对样本数为1的数据进行增强
    #         length = len(train_data_dict[key])
    #         if length < sample_threshold:
    #             delta_len = sample_threshold - length   # 针对某种病种组合，需要生成的样本的数量
    #
    #             ecg_data_list = []
    #             for patient in train_data_dict[key]:
    #                 file_path = os.path.join(args.train_data_dir, patient + ".mat")
    #                 ecg_data = sio.loadmat(file_path)['ecgdata']
    #                 ecg_data_list.append(ecg_data)
    #
    #             # 将该病种组合中的样本拼接为一个大矩阵
    #             sample_data = ecg_data_list[0]
    #             for i in range(len(ecg_data_list)):
    #                 if i + 1 < len(ecg_data_list):      # 第一张样本已经赋值给sample_data了，因此从第二张样本开始拼接
    #                     sample_data = np.hstack((sample_data, ecg_data_list[i+1]))      # 样本拼接
    #
    #             # 从sample_data中采样生成delta_len个样本
    #             for j in range(delta_len):
    #                 # 随机生成从矩阵中采样的起点
    #                 start = np.random.RandomState(args.seed).randint(0, 4999)
    #                 end = start + 5000
    #                 sample_data_len = sample_data.shape[1]
    #
    #                 if end > sample_data_len:
    #                     # 通过两段合成为一个样本
    #                     s1_len = sample_data_len - start            # 第一段样本的长度
    #                     s2_len = 5000 - s1_len                      # # 第二段样本的长度
    #                     sample1 = sample_data[:, start:]
    #                     sample2 = sample_data[:, :s2_len]
    #                     sample = np.hstack((sample1, sample2))
    #                 else:
    #                     sample = sample_data[:, start:end]
    #
    #                 # 保存生成的样本和样本对应的病种组合
    #                 patient_id = "TEST" + str(generate_num)
    #                 mat_path = os.path.join(generate_sample_dir, patient_id + ".mat")
    #                 sio.savemat(mat_path, {'ecgdata': sample})
    #
    #                 generate_sample_dict[patient_id] = key
    #                 generate_num += 1

    # 4.1 拼接生成的样本的patient_id和对应的病种，便于写入到文件中
    result_sample = []
    for key in generate_sample_dict.keys():
        patient_id = key
        res = str(patient_id)
        disease = generate_sample_dict[key].split("_")
        for dis in disease:
            res = res + "," + dis
        result_sample.append(res)
    random.shuffle(result_sample)

    # 5. 获取原始的trainreference数据中的patient_id和对应的病种
    result = []
    df_org = pd.read_csv(org_reference_path, sep='\t')
    for _, rows in df_org.iterrows():
        rows = rows[0].split(',')
        res = rows[0]
        patient_id = rows[0]
        if patient_id not in conflict_id:
            for disease in rows[1:]:
                res = res + "," + disease
            result.append(res)

    # 将原始数据的patient_id及病种和生成的样本的patient_id及病种合并写入文件
    result += result_sample
    with open(new_reference_path, "w", encoding="utf-8") as fw:
        for res in result:
            fw.write(res)
            fw.write('\n')
        fw.close()

    logger.info("generate sample: %d", generate_num - 30000)

    # 将测试数据的patient_id保存到test_data_txt文件中
    with open(args.test_data_txt, "w", encoding="utf-8") as fw:
        for name in test_data:
            fw.write(name)
            fw.write("\n")
        fw.close()

    # 将测试数据的patient_id保存到dev_data_txt文件中
    with open(args.dev_data_txt, "w", encoding="utf-8") as fw:
        for name in dev_data:
            fw.write(name)
            fw.write("\n")
        fw.close()

    # 将用于寻找lgb的最佳阈值的样本的patient_id保存到lgb_threshold_data_txt文件中
    with open(args.lgb_threshold_data_txt, "w", encoding="utf-8") as fw:
        for name in lgb_threshold_data:
            fw.write(name)
            fw.write("\n")
        fw.close()


# 从所有样本中采样训练集和验证集（没有测试集）
def split_train_dev_data():
    # 第一步：采样验证集dev data，剩下的数据作为训练集
    # 第二步：数据增强，对小样本数据进行重采样，生成部分样本

    logger.info("split train test data start...")
    generate_sample_dir = args.train_data_dir  # todo: 保存生成样本的路径

    conflict_id = get_conflict_samples()
    conflict_id += bad_list  # 将病种有冲突的和图形效果很差的样本合并到一起，从数据集中去除
    disease_dict = dict()  # 保存所有的数据，key：病种编号组合出的字符串，value:patient id构成的list
    dev_reat = 0.2  # 验证集占训练集和验证集总数的比例
    dev_sample_thresholds = 5  # 采样阈值：从样本数大于5个的样本中采样验证集
    dev_data = list()  # 保存验证集样本的patient_id
    train_data_dict = dict()  # 保存去除验证集后的数据，作为训练集
    mini_sample_threshold = 1  # 小样本数量阈值
    mini_sample_patient = []  # 统计病种组合对应的样本数为1的样本的patient_id
    mini_sample_key = []  # 统计病种组合对应的样本数为1的病种
    sample_threshold = 20  # 采样生成样本的阈值：经测试，样本增强到20效果最佳
    generate_num = 30000  # 生成样本的起始patient_id

    # 统计病种的组合种类，及每个种类的数量
    df_reference = pd.read_csv(args.trainreference, sep='\t', header=None)
    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        # 将样本的病种合并为字符串，作为字典的key
        if patient_id not in conflict_id:
            key = ""
            for dis in disease:
                if dis != disease[-1]:
                    key += dis + "_"
                else:
                    key += dis

        # 将样本的patient_id构成的list作为字典的value
        if key not in disease_dict.keys():
            value = list()
            value.append(patient_id)
            disease_dict[key] = value
        else:
            disease_dict[key].append(patient_id)

    # 将病种组合写入文件，用于对预测的结果进行校正
    with open(args.disease_groups, "w", encoding='utf-8') as fw1:
        for key in disease_dict.keys():
            fw1.write(key)
            fw1.write('\n')
        fw1.close()

    # 统计样本数为1的样本对应的patient_id
    for key in disease_dict.keys():
        if len(disease_dict[key]) <= mini_sample_threshold:  # todo
            mini_sample_patient += disease_dict[key]
            mini_sample_key.append(key)

    with open(args.mini_sample_txt, "w", encoding="utf-8") as fw2:
        for sample in mini_sample_patient:
            fw2.write(sample)
            fw2.write("\n")
        fw2.close()

    # 第一步：采样出验证集，剩下的作为训练集
    for key in disease_dict.keys():
        if len(disease_dict[key]) >= dev_sample_thresholds:
            dev_len = math.floor(dev_reat * len(disease_dict[key])) + 1  # 保证至少采样一个样本作为测试集
            random.shuffle(disease_dict[key])
            dev_sample = disease_dict[key][:dev_len]  # 获取测试集数据
            dev_data += dev_sample
            train_sample = disease_dict[key][dev_len:]  # 剩余部分作为训练集和验证集
            train_data_dict[key] = train_sample
        else:
            train_data_dict[key] = disease_dict[key]

    # 第二步：数据增强，生成小样本数据
    # 3.1 对训练样本进行处理，如果train_data_dict中每种组合病种的样本数量少于20，则从该类组合病种的样本中采样，生成20个样本
    generate_sample_dict = dict()  # key:生成的样本的patient_id，value：病种的组合

    # for key in train_data_dict.keys():
    #     if key not in mini_sample_key:      # 不对样本数为1的数据进行增强
    #         length = len(train_data_dict[key])
    #         if length < sample_threshold:
    #             delta_len = sample_threshold - length   # 针对某种病种组合，需要生成的样本的数量
    #
    #             ecg_data_list = []
    #             for patient in train_data_dict[key]:
    #                 file_path = os.path.join(args.train_data_dir, patient + ".mat")
    #                 ecg_data = sio.loadmat(file_path)['ecgdata']
    #                 ecg_data_list.append(ecg_data)
    #
    #             # 将该病种组合中的样本拼接为一个大矩阵
    #             sample_data = ecg_data_list[0]
    #             for i in range(len(ecg_data_list)):
    #                 if i + 1 < len(ecg_data_list):      # 第一张样本已经赋值给sample_data了，因此从第二张样本开始拼接
    #                     sample_data = np.hstack((sample_data, ecg_data_list[i+1]))      # 样本拼接
    #
    #             # 从sample_data中采样生成delta_len个样本
    #             for j in range(delta_len):
    #                 # 随机生成从矩阵中采样的起点
    #                 start = np.random.RandomState(args.seed).randint(0, 4999)
    #                 end = start + 5000
    #                 sample_data_len = sample_data.shape[1]
    #
    #                 if end > sample_data_len:
    #                     # 通过两段合成为一个样本
    #                     s1_len = sample_data_len - start            # 第一段样本的长度
    #                     s2_len = 5000 - s1_len                      # # 第二段样本的长度
    #                     sample1 = sample_data[:, start:]
    #                     sample2 = sample_data[:, :s2_len]
    #                     sample = np.hstack((sample1, sample2))
    #                 else:
    #                     sample = sample_data[:, start:end]
    #
    #                 # 保存生成的样本和样本对应的病种组合
    #                 patient_id = "TEST" + str(generate_num)
    #                 mat_path = os.path.join(generate_sample_dir, patient_id + ".mat")
    #                 sio.savemat(mat_path, {'ecgdata': sample})
    #
    #                 generate_sample_dict[patient_id] = key
    #                 generate_num += 1

    # 3.1 拼接生成的样本的patient_id和对应的病种，便于写入到文件中
    result_sample = []
    for key in generate_sample_dict.keys():
        patient_id = key
        res = str(patient_id)
        disease = generate_sample_dict[key].split("_")
        for dis in disease:
            res = res + "," + dis
        result_sample.append(res)
    random.shuffle(result_sample)

    # 3.2 获取原始的trainreference数据中的patient_id和对应的病种
    result = []
    df_org = pd.read_csv(args.trainreference, sep='\t')
    for _, rows in df_org.iterrows():
        rows = rows[0].split(',')
        res = rows[0]
        patient_id = rows[0]
        if patient_id not in conflict_id:
            for disease in rows[1:]:
                res = res + "," + disease
            result.append(res)

    # 3.3 将原始数据的patient_id及病种和生成的样本的patient_id及病种合并写入文件
    result += result_sample
    with open(args.trainreference_new, "w", encoding="utf-8") as fw:
        for res in result:
            fw.write(res)
            fw.write('\n')
        fw.close()

    logger.info("generate sample: %d", generate_num - 30000)

    # 将测试数据的文件名保存到dev_data_txt文件中
    with open(args.dev_data_txt, "w", encoding="utf-8") as fw:
        for name in dev_data:
            fw.write(name)
            fw.write("\n")
        fw.close()


# 获取训练others模型的训练集、验证集和测试集
def split_train_dev_test_data_others():
    '''
    将所有含有12号病的都作为正样本。共计1985个样本。
    从normal中抽取1985个样本。
    :return:
    '''

    # 第一步：采样测试集test data
    # 第二步：采样验证集dev data
    # 第三步：剩下的数据作为训练集，并对小样本数据进行重采样，生成部分样本

    logger.info("split train test data start...")
    org_reference_path = args.trainreference
    other_reference_path = args.trainreference_others
    mini_sample_file_path = args.mini_sample_other_txt

    conflict_id = get_conflict_samples()
    conflict_id += bad_list  # 将病种有冲突的和图形效果很差的样本合并到一起，从数据集中去除
    disease_dict = dict()  # 保存所有的数据，key：病种编号组合出的字符串，value:patient id构成的list
    test_reat = 0.1  # 测试集占整个数据集的比例
    test_sample_thresholds = 10  # 采样阈值：从样本数大于10个的样本中采样测试集
    test_data = list()  # 保存测试集样本的patient_id
    train_dev_data_dict = dict()  # 保存去除测试集后的数据，作为训练集和验证集
    dev_reat = 0.2  # 验证集占训练集和验证集总数的比例
    dev_sample_thresholds = 5  # 采样阈值：从样本数大于5个的样本中采样验证集
    dev_data = list()  # 保存验证集样本的patient_id
    train_data_dict = dict()  # 保存去除验证集后的数据，作为训练集
    mini_sample_threshold = 1  # 小样本数阈值
    mini_sample_patient = []  # 统计病种组合对应的样本数为1的样本的patient_id
    mini_sample_key = []  # 统计病种组合对应的样本数为1的病种
    normal_rate = 0.273  # 从normal样本中采样的比例

    # 统计原始数据中病种的组合种类，及每个种类的数量
    df_reference = pd.read_csv(org_reference_path, sep='\t', header=None)
    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        # 将样本的病种合并为字符串，作为字典的key
        if patient_id not in conflict_id:
            key = ""
            for dis in disease:
                if dis != disease[-1]:
                    key += dis + "_"
                else:
                    key += dis

        # 将样本的patient_id构成的list作为字典的value
        if key not in disease_dict.keys():
            value = list()
            value.append(patient_id)
            disease_dict[key] = value
        else:
            disease_dict[key].append(patient_id)

    # 将病种组合写入文件，用于对预测的结果进行校正
    with open(args.disease_groups, "w", encoding='utf-8') as fw1:  # todo
        for key in disease_dict.keys():
            fw1.write(key)
            fw1.write('\n')
        fw1.close()

    # 统计样本数为1的样本对应的patient_id
    for key in disease_dict.keys():
        if len(disease_dict[key]) <= mini_sample_threshold:
            mini_sample_patient += disease_dict[key]
            mini_sample_key.append(key)

    with open(mini_sample_file_path, "w", encoding="utf-8") as fw2:
        for sample in mini_sample_patient:
            fw2.write(sample)
            fw2.write("\n")
        fw2.close()

    # 获取训练others二分类模型的总体数据集（包含训练集、验证集、测试集）
    others_samples_dict = dict()  # 保存用于训练others模型的训练集、验证集和测试集
    for key in disease_dict.keys():
        if key == str(1):  # 获取Normal样本，并采样
            normal_list = disease_dict[key]
            normal_len = math.floor(len(normal_list) * normal_rate)
            random.shuffle(normal_list)
            normal_samples = normal_list[:normal_len]
            others_samples_dict[key] = normal_samples

        if str(12) in key:  # 获取包含12号病的样本
            others_samples_dict[key] = disease_dict[key]

    # 将采样得到的用于训练others模型的数据的patient_id和病种写入到reference.csv中
    result_sample = []
    for key in others_samples_dict.keys():
        disease = key.split("_")
        patient_id_list = others_samples_dict[key]
        for patient in patient_id_list:
            res = str(patient)
            for dis in disease:
                res = res + "," + dis
            result_sample.append(res)
    random.shuffle(result_sample)

    # 将样本的patient_id及病种写入文件
    with open(other_reference_path, "w", encoding="utf-8") as fw:  # todo
        for res in result_sample:
            fw.write(res)
            fw.write('\n')
        fw.close()

    # 第一步：采样出测试集，剩下的作为训练集和验证集
    for key in others_samples_dict.keys():
        if len(others_samples_dict[key]) >= test_sample_thresholds:
            test_len = math.floor(test_reat * len(others_samples_dict[key])) + 1  # 保证至少采样一个样本作为测试集
            random.shuffle(others_samples_dict[key])
            test_sample = others_samples_dict[key][:test_len]  # 获取测试集数据
            test_data += test_sample
            train_dev_sample = others_samples_dict[key][test_len:]  # 剩余部分作为训练集和验证集
            train_dev_data_dict[key] = train_dev_sample
        else:
            train_dev_data_dict[key] = others_samples_dict[key]

    # 第二步：采样出验证集，剩下的作为训练集
    for key in train_dev_data_dict.keys():
        if len(train_dev_data_dict[key]) >= dev_sample_thresholds:
            dev_len = math.floor(dev_reat * len(train_dev_data_dict[key])) + 1  # 保证至少采样一个样本作为验证集
            random.shuffle(train_dev_data_dict[key])
            dev_sample = train_dev_data_dict[key][:dev_len]  # 获取测试集数据
            dev_data += dev_sample
            train_sample = train_dev_data_dict[key][dev_len:]  # 剩余部分作为训练集
            train_data_dict[key] = train_sample
        else:
            train_data_dict[key] = train_dev_data_dict[key]

    # 将测试数据集的patient_id保存到test_data_txt文件中
    with open(args.test_data_other_txt, "w", encoding="utf-8") as fw:
        for name in test_data:
            fw.write(name)
            fw.write("\n")
        fw.close()

    # 将验证集的patient_id保存到dev_data_txt文件中
    with open(args.dev_data_other_txt, "w", encoding="utf-8") as fw:
        for name in dev_data:
            fw.write(name)
            fw.write("\n")
        fw.close()


# 获取训练others模型的训练集、验证集(没有测试集)
def split_train_dev_data_others():
    '''
    将所有含有12号病的都作为正样本。共计1985个样本。
    从normal中抽取1985个样本。
    :return:
    '''

    # 第一步：采样测试集test data
    # 第二步：采样验证集dev data
    # 第三步：剩下的数据作为训练集，并对小样本数据进行重采样，生成部分样本

    logger.info("split train test data start...")
    org_reference_path = args.trainreference
    other_reference_path = args.trainreference_others
    mini_sample_file_path = args.mini_sample_other_txt

    conflict_id = get_conflict_samples()
    conflict_id += bad_list  # 将病种有冲突的和图形效果很差的样本合并到一起，从数据集中去除
    disease_dict = dict()  # 保存所有的数据，key：病种编号组合出的字符串，value:patient id构成的list
    dev_reat = 0.2  # 验证集占训练集和验证集总数的比例
    dev_sample_thresholds = 5  # 采样阈值：从样本数大于5个的样本中采样验证集
    dev_data = list()  # 保存验证集样本的patient_id
    train_data_dict = dict()  # 保存去除验证集后的数据，作为训练集
    mini_sample_threshold = 1  # 小样本数阈值
    mini_sample_patient = []  # 统计病种组合对应的样本数为1的样本的patient_id
    mini_sample_key = []  # 统计病种组合对应的样本数为1的病种
    normal_rate = 0.273  # 从normal样本中采样的比例

    # 统计原始数据中病种的组合种类，及每个种类的数量
    df_reference = pd.read_csv(org_reference_path, sep='\t', header=None)
    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        # 将样本的病种合并为字符串，作为字典的key
        if patient_id not in conflict_id:
            key = ""
            for dis in disease:
                if dis != disease[-1]:
                    key += dis + "_"
                else:
                    key += dis

        # 将样本的patient_id构成的list作为字典的value
        if key not in disease_dict.keys():
            value = list()
            value.append(patient_id)
            disease_dict[key] = value
        else:
            disease_dict[key].append(patient_id)

    # 将病种组合写入文件，用于对预测的结果进行校正
    with open(args.disease_groups, "w", encoding='utf-8') as fw1:  # todo
        for key in disease_dict.keys():
            fw1.write(key)
            fw1.write('\n')
        fw1.close()

    # 统计样本数为1的样本对应的patient_id
    for key in disease_dict.keys():
        if len(disease_dict[key]) <= mini_sample_threshold:
            mini_sample_patient += disease_dict[key]
            mini_sample_key.append(key)

    with open(mini_sample_file_path, "w", encoding="utf-8") as fw2:
        for sample in mini_sample_patient:
            fw2.write(sample)
            fw2.write("\n")
        fw2.close()

    # 获取训练others二分类模型的总体数据集（包含训练集、验证集、测试集）
    others_samples_dict = dict()  # 保存用于训练others模型的训练集、验证集和测试集
    for key in disease_dict.keys():
        if key == str(1):  # 获取Normal样本，并采样
            normal_list = disease_dict[key]
            normal_len = math.floor(len(normal_list) * normal_rate)
            random.shuffle(normal_list)
            normal_samples = normal_list[:normal_len]
            others_samples_dict[key] = normal_samples

        if str(12) in key:  # 获取包含12号病的样本
            others_samples_dict[key] = disease_dict[key]

    # 将采样得到的用于训练others模型的数据的patient_id和病种写入到reference.csv中
    result_sample = []
    for key in others_samples_dict.keys():
        disease = key.split("_")
        patient_id_list = others_samples_dict[key]
        for patient in patient_id_list:
            res = str(patient)
            for dis in disease:
                res = res + "," + dis
            result_sample.append(res)
    random.shuffle(result_sample)

    # 将样本的patient_id及病种写入文件
    with open(other_reference_path, "w", encoding="utf-8") as fw:  # todo
        for res in result_sample:
            fw.write(res)
            fw.write('\n')
        fw.close()

    # 第一步：采样出验证集，剩下的作为训练集
    for key in others_samples_dict.keys():
        if len(others_samples_dict[key]) >= dev_sample_thresholds:
            dev_len = math.floor(dev_reat * len(others_samples_dict[key])) + 1  # 保证至少采样一个样本作为测试集
            random.shuffle(others_samples_dict[key])
            dev_sample = others_samples_dict[key][:dev_len]  # 获取测试集数据
            dev_data += dev_sample
            train_sample = others_samples_dict[key][dev_len:]  # 剩余部分作为训练集和验证集
            train_data_dict[key] = train_sample
        else:
            train_data_dict[key] = others_samples_dict[key]

    # 将验证集的patient_id保存到dev_data_txt文件中
    with open(args.dev_data_other_txt, "w", encoding="utf-8") as fw:
        for name in dev_data:
            fw.write(name)
            fw.write("\n")
        fw.close()


# 从所有样本中采样训练normal模型的训练集、验证集、测试集
def split_train_dev_test_data_normal():
    # 第一步：采样测试集test data
    # 第二步：采样验证集dev data， 剩下的数据作为训练集train data

    logger.info("split train test data start...")
    org_reference_path = args.trainreference
    normal_reference_path = args.trainreference_normal
    mini_sample_file_path = args.mini_sample_normal_txt

    conflict_id = get_conflict_samples()
    conflict_id += bad_list  # 将病种有冲突的和图形效果很差的样本合并到一起，从数据集中去除
    disease_dict = dict()  # 保存所有的数据，key：病种编号组合出的字符串，value:patient id构成的list
    test_reat = 0.1  # 测试集占整个数据集的比例
    test_sample_thresholds = 10  # 采样阈值：从样本数大于10个的样本中采样测试集
    test_data = list()  # 保存测试集样本的patient_id
    train_dev_data_dict = dict()  # 保存去除测试集后的数据，作为训练集和验证集
    dev_reat = 0.2  # 验证集占训练集和验证集总数的比例
    dev_sample_thresholds = 5  # 采样阈值：从样本数大于5个的样本中采样验证集
    dev_data = list()  # 保存验证集样本的patient_id
    train_data_dict = dict()  # 保存去除验证集后的数据，作为训练集
    mini_sample_threshold = 2  # 小样本数阈值
    mini_sample_patient = []  # 统计病种组合对应的样本数为1的样本的patient_id
    mini_sample_key = []  # 统计病种组合对应的样本数为1的病种

    # 统计原始数据中病种的组合种类，及每个种类的数量
    df_reference = pd.read_csv(org_reference_path, sep='\t', header=None)
    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        # 将样本的病种合并为字符串，作为字典的key
        if patient_id not in conflict_id:
            key = ""
            for dis in disease:
                if dis != disease[-1]:
                    key += dis + "_"
                else:
                    key += dis

        # 将样本的patient_id构成的list作为字典的value
        if key not in disease_dict.keys():
            value = list()
            value.append(patient_id)
            disease_dict[key] = value
        else:
            disease_dict[key].append(patient_id)

    # 将病种组合写入文件，用于对预测的结果进行校正
    with open(args.disease_groups, "w", encoding='utf-8') as fw1:
        for key in disease_dict.keys():
            fw1.write(key)
            fw1.write('\n')
        fw1.close()

    # 统计样本数为1的样本对应的patient_id
    for key in disease_dict.keys():
        if len(disease_dict[key]) <= mini_sample_threshold:  # todo
            mini_sample_patient += disease_dict[key]
            mini_sample_key.append(key)

    with open(mini_sample_file_path, "w", encoding="utf-8") as fw2:
        for sample in mini_sample_patient:
            fw2.write(sample)
            fw2.write("\n")
        fw2.close()

    # 第一步：采样出测试集，剩下的作为训练集和验证集
    for key in disease_dict.keys():
        if len(disease_dict[key]) >= test_sample_thresholds:
            test_len = math.floor(test_reat * len(disease_dict[key])) + 1  # 保证至少采样一个样本作为测试集
            random.shuffle(disease_dict[key])
            test_sample = disease_dict[key][:test_len]  # 获取测试集数据
            test_data += test_sample
            train_dev_sample = disease_dict[key][test_len:]  # 剩余部分作为训练集和验证集
            train_dev_data_dict[key] = train_dev_sample
        else:
            train_dev_data_dict[key] = disease_dict[key]

    # 第二步：采样出验证集，剩下的作为训练集
    for key in train_dev_data_dict.keys():
        if len(train_dev_data_dict[key]) >= dev_sample_thresholds:
            dev_len = math.floor(dev_reat * len(train_dev_data_dict[key])) + 1  # 保证至少采样一个样本作为验证集
            random.shuffle(train_dev_data_dict[key])
            dev_sample = train_dev_data_dict[key][:dev_len]  # 获取测试集数据
            dev_data += dev_sample
            train_sample = train_dev_data_dict[key][dev_len:]  # 剩余部分作为训练集
            train_data_dict[key] = train_sample
        else:
            train_data_dict[key] = train_dev_data_dict[key]

    # 3. 获取原始的trainreference数据中的patient_id和对应的病种
    result = []
    df_org = pd.read_csv(org_reference_path, sep='\t')
    for _, rows in df_org.iterrows():
        rows = rows[0].split(',')
        res = rows[0]
        patient_id = rows[0]
        if patient_id not in conflict_id:
            for disease in rows[1:]:
                res = res + "," + disease
            result.append(res)

    # 将原始数据去除冲突样本后的的patient_id写入文件
    with open(normal_reference_path, "w", encoding="utf-8") as fw:
        for res in result:
            fw.write(res)
            fw.write('\n')
        fw.close()

    # 将测试数据的patient_id保存到test_data_txt文件中
    with open(args.test_data_normal_txt, "w", encoding="utf-8") as fw:
        for name in test_data:
            fw.write(name)
            fw.write("\n")
        fw.close()

    # 将测试数据的patient_id保存到dev_data_txt文件中
    with open(args.dev_data_normal_txt, "w", encoding="utf-8") as fw:
        for name in dev_data:
            fw.write(name)
            fw.write("\n")
        fw.close()


# 从所有样本中采样训练normal模型的训练集和验证集（没有测试集）
def split_train_dev_data_normal():
    # 第一步：采样验证集dev data，剩下的数据作为训练集

    logger.info("split train test data start...")
    org_reference_path = args.trainreference
    normal_reference_path = args.trainreference_normal
    mini_sample_file_path = args.mini_sample_normal_txt

    conflict_id = get_conflict_samples()
    conflict_id += bad_list  # 将病种有冲突的和图形效果很差的样本合并到一起，从数据集中去除
    disease_dict = dict()  # 保存所有的数据，key：病种编号组合出的字符串，value:patient id构成的list
    dev_reat = 0.2  # 验证集占训练集和验证集总数的比例
    dev_sample_thresholds = 5  # 采样阈值：从样本数大于5个的样本中采样验证集
    dev_data = list()  # 保存验证集样本的patient_id
    train_data_dict = dict()  # 保存去除验证集后的数据，作为训练集
    mini_sample_threshold = 2  # 小样本数量阈值
    mini_sample_patient = []  # 统计病种组合对应的样本数为1的样本的patient_id
    mini_sample_key = []  # 统计病种组合对应的样本数为1的病种

    # 统计病种的组合种类，及每个种类的数量
    df_reference = pd.read_csv(org_reference_path, sep='\t', header=None)
    for _, rows in df_reference.iterrows():
        info = rows[0].split(",")
        patient_id = info[0]
        disease = info[1:]

        # 将样本的病种合并为字符串，作为字典的key
        if patient_id not in conflict_id:
            key = ""
            for dis in disease:
                if dis != disease[-1]:
                    key += dis + "_"
                else:
                    key += dis

        # 将样本的patient_id构成的list作为字典的value
        if key not in disease_dict.keys():
            value = list()
            value.append(patient_id)
            disease_dict[key] = value
        else:
            disease_dict[key].append(patient_id)

    # 将病种组合写入文件，用于对预测的结果进行校正
    with open(args.disease_groups, "w", encoding='utf-8') as fw1:
        for key in disease_dict.keys():
            fw1.write(key)
            fw1.write('\n')
        fw1.close()

    # 统计样本数为1的样本对应的patient_id
    for key in disease_dict.keys():
        if len(disease_dict[key]) <= mini_sample_threshold:  # todo
            mini_sample_patient += disease_dict[key]
            mini_sample_key.append(key)

    with open(mini_sample_file_path, "w", encoding="utf-8") as fw2:
        for sample in mini_sample_patient:
            fw2.write(sample)
            fw2.write("\n")
        fw2.close()

    # 第一步：采样出验证集，剩下的作为训练集
    for key in disease_dict.keys():
        if len(disease_dict[key]) >= dev_sample_thresholds:
            dev_len = math.floor(dev_reat * len(disease_dict[key])) + 1  # 保证至少采样一个样本作为测试集
            random.shuffle(disease_dict[key])
            dev_sample = disease_dict[key][:dev_len]  # 获取测试集数据
            dev_data += dev_sample
            train_sample = disease_dict[key][dev_len:]  # 剩余部分作为训练集和验证集
            train_data_dict[key] = train_sample
        else:
            train_data_dict[key] = disease_dict[key]

    # 3.2 获取原始的trainreference数据中的patient_id和对应的病种
    result = []
    df_org = pd.read_csv(args.trainreference, sep='\t')
    for _, rows in df_org.iterrows():
        rows = rows[0].split(',')
        res = rows[0]
        patient_id = rows[0]
        if patient_id not in conflict_id:
            for disease in rows[1:]:
                res = res + "," + disease
            result.append(res)

    # 3.3 将原始数据的patient_id及病种和生成的样本的patient_id及病种合并写入文件
    with open(normal_reference_path, "w", encoding="utf-8") as fw:
        for res in result:
            fw.write(res)
            fw.write('\n')
        fw.close()

    # 将测试数据的文件名保存到dev_data_txt文件中
    with open(args.dev_data_normal_txt, "w", encoding="utf-8") as fw:
        for name in dev_data:
            fw.write(name)
            fw.write("\n")
        fw.close()

