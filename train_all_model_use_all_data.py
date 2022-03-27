# -*- coding: utf-8 -*-
# @Time    : 2021/12/22 22:33
# @Author  : Qinglong

'''使用所有的数据训练模型，不保留测试集'''

from utils.tools import logger

from train_resnet_1d_3_5 import train_model as train_model1
from train_resnet_1d_7_15 import train_model as train_model2
from train_resnet_2d_3_5 import train_model as train_model3
from train_resnet_2d_7_15 import train_model as train_model4
from train_lgb_2_model import train_lgb_model as train_model5
from train_resnet_1d_others import train_model as train_model6
from train_resnet_2d_others import train_model as train_model7
from train_lgb_2_model_others import train_lgb_model as train_model8
from train_lgb_4_model import train_lgb_model as train_model9
from train_resnet_1d_normal import train_model as train_model10
from train_resnet_2d_normal import train_model as train_model11

from utils.preprocess import split_train_dev_data, get_train_dev_label_local
from utils.preprocess import split_train_dev_test_data, get_train_dev_test_label_local
from utils.preprocess import split_train_dev_data_others, get_train_dev_label_others_local
from utils.preprocess import split_train_dev_data_normal, get_train_dev_label_normal_local


logger.info("train_resnet_1d_3_5")
split_train_dev_data()
get_train_dev_label_local()             # 将所有数据划分为训练集和验证集，没有测试集
# split_train_dev_test_data()
# get_train_dev_test_label_local()      # 将所有数据划分为训练集、验证集和测试集
train_model1()

logger.info("train_resnet_1d_7_15")
split_train_dev_data()
get_train_dev_label_local()             # 将所有数据划分为训练集和验证集，没有测试集
# split_train_dev_test_data()
# get_train_dev_test_label_local()      # 将所有数据划分为训练集、验证集和测试集
train_model2()

logger.info("train_resnet_2d_3_5")
split_train_dev_data()
get_train_dev_label_local()             # 将所有数据划分为训练集和验证集，没有测试集
# split_train_dev_test_data()
# get_train_dev_test_label_local()      # 将所有数据划分为训练集、验证集和测试集
train_model3()

logger.info("train_resnet_2d_7_15")
split_train_dev_data()
get_train_dev_label_local()             # 将所有数据划分为训练集和验证集，没有测试集
# split_train_dev_test_data()
# get_train_dev_test_label_local()      # 将所有数据划分为训练集、验证集和测试集
train_model4()

logger.info("train_lgb_2_model")
train_model5()


# 训练识别others和normal的模型
logger.info("train_resnet_1d_others")
split_train_dev_data_others()
get_train_dev_label_others_local()
train_model6()


logger.info("train_resnet_2d_others")
split_train_dev_data_others()
get_train_dev_label_others_local()
train_model7()


logger.info("train_lgb_2_model_others")
train_model8()

logger.info("train_lgb_4_model")
train_model9()



logger.info("train_resnet_1d_normal")
split_train_dev_data_normal()
get_train_dev_label_normal_local()
train_model10()


logger.info("train_resnet_2d_normal")
split_train_dev_data_normal()
get_train_dev_label_normal_local()
train_model11()
