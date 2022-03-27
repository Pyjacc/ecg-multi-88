# _*_ coding:utf-8 _*_


'''使用resnet34_1d和resnet34_2d共计4个模型抽取的feature训练lightgbm模型'''

import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
import joblib
from utils.tools import tools
from utils.tools import logger
from utils.configures import args
from utils.features import GetFeature
from models.resnet.resnet34_rdrop import resnet341d3_5, resnet341d7_15, resnet342d3_5, resnet342d7_15     # todo


warnings.filterwarnings('ignore', category=FutureWarning)

# get optimal thresholds
def get_threshold(model, data, label, threshold_file_path):
    logger.info("get optimal thresholds, please wait...")
    thresholds = []
    y_val_scores = model.predict_proba(data)
    for i in range(len(args.ecg_classes)):
        y_val_score = y_val_scores[:, i]
        threshold = tools.find_optimal_threshold(label[:, i], y_val_score)
        thresholds.append(threshold)

    logger.info("write optimal thresholds to file: %s", threshold_file_path)
    with open(threshold_file_path, "w", encoding="utf-8") as fw:        # todo: 需要根据使用的模型不同，将阈值写入相应的文件
        for threshold in thresholds:
            fw.write(str(threshold))
            fw.write("\n")
        fw.close()

    logger.info("thresholds: %s" % thresholds)


def get_feature(model_1d_path1, model_1d_path2, model_2d_path1, model_2d_path2, feature_path):
    logger.info("get feature")
    args.get_feature = True
    # 使用resnet模型抽取所有训练集的特征，并保存到csv文件
    GetFeature().get_feature_from_4_resnet(model_1d_path1, model_1d_path2, model_2d_path1, model_2d_path2,
                                           resnet341d7_15, resnet341d3_5, resnet342d7_15, resnet342d3_5,
                                           feature_path)
    args.get_feature = False


def train_lgb_model():
    model_1d_path_7_15 = args.model_path_1d_7_15          # todo: 需要修改的变量
    model_1d_path_3_5 = args.model_path_1d_3_5
    model_2d_path_7_15 = args.model_path_2d_7_15
    model_2d_path_3_5 = args.model_path_2d_3_5
    feature_path = args.feature_renet_4_model_csv
    lgb_model_path = args.lgb_4_model_path
    threshold_file_path = args.threshold_4_model_lgb

    get_feature(model_1d_path_7_15, model_1d_path_3_5, model_2d_path_7_15, model_2d_path_3_5, feature_path)

    train_random_num, dev_random_num = tools.split_train_dev_data(seed=args.seed)
    df = pd.read_csv(feature_path, sep='\t')          # feature文件路径要根据使用的不同的feature对应的文件而定
    train_data, dev_data = [], []
    train_lable, dev_label = [], []

    logger.info("get train data and labels start")
    for _, rows in df.iterrows():
        rows = rows[0].split(",")
        label = rows[:12]
        data = rows[12:-1]
        random_num = rows[-1]

        for i in range(len(label)):
            label[i] = int(float(label[i]))  # 将label转为0/1整数

        if int(random_num) in train_random_num:
            train_lable.append(label)
            train_data.append(data)
        else:
            dev_label.append(label)
            dev_data.append(data)

    logger.info("get train data and labels success")
    train_data, dev_data = np.array(train_data), np.array(dev_data)
    train_lable, dev_label = np.array(train_lable), np.array(dev_label)
    logger.info("train data shape: %s, train lable shape: %s" % (train_data.shape, train_lable.shape))

    logger.info("train lightgbm model start, please wait...")
    model = LGBMClassifier(n_estimators=100, seed=args.seed)
    model = OneVsRestClassifier(model)      # sklearn提供的multiclass子类OneVsRestClassifier实现多标签分类
    model.fit(train_data, train_lable)
    logger.info("train lightgbm model end, save model: %s", lgb_model_path)
    joblib.dump(model, lgb_model_path)     # todo: 需要根据使用的模型不同，设置不同的模型保存路径
    logger.info("save lightgbm model success")

    # get optimal thresholds
    model = joblib.load(lgb_model_path)            # 使用已经训练好的lgb模型寻找阈值
    get_threshold(model, dev_data, dev_label, threshold_file_path)
    logger.info("train lgb model success!")


if __name__ == "__main__":
    logger.info("train model")
    train_lgb_model()

