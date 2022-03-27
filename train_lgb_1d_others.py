# _*_ coding:utf-8 _*_


'''
(1)使用resnet34抽取的feature训练lightgbm模型
(2)用于判断心电图是others还是正常
'''


import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import joblib
from utils.tools import tools
from utils.tools import logger
from utils.configures import args
from steps.other.feature_other import GetFeature        # todo： 根据不同的模型选择GetFeature
from models.resnet.resnet34_rdrop import resnet341d7_15

warnings.filterwarnings('ignore', category=FutureWarning)

# get optimal thresholds
def get_threshold(model, data, label):
    logger.info("get optimal thresholds, please wait...")
    thresholds = []
    y_val_scores = model.predict(data)

    for i in range(len(args.ecg_classes_others)):      # todo: 根据不同的模型选择ecg_classes
        y_val_score = y_val_scores[:, i]
        threshold = tools.find_optimal_threshold(label[:, i], y_val_score)
        thresholds.append(threshold)

    logger.info("write optimal thresholds to file")
    with open(args.threshold_lgb_others, "w", encoding="utf-8") as fw:        # todo: 需要根据使用的模型不同，将阈值写入相应的文件
        for threshold in thresholds:
            fw.write(str(threshold))
            fw.write("\n")
        fw.close()

    logger.info("thresholds: %s" % thresholds)


def get_feature(model_1d_path, feature_path):
    logger.info("get feature")
    args.get_feature = True
    GetFeature().get_feature_from_resnet(model_1d_path, resnet341d7_15, feature_path)        # 使用resnet模型抽取所有训练集的特征，并保存到csv文件
    args.get_feature = False


def train_lgb_model():
    threshold_file_path = args.threshold_lgb_others
    model_1d_path = args.model_path_others_1d
    lgb_model_path = args.lgb_others_1d_path
    feature_path = args.feature_renet_others_csv

    get_feature(model_1d_path, feature_path)

    df = pd.read_csv(feature_path, sep='\t')          # todo:feature文件路径要根据使用的不同的feature对应的文件而定
    train_data, train_lable = [], []

    logger.info("get train data and labels start")
    for _, rows in df.iterrows():
        rows = rows[0].split(",")
        label = rows[:1]        # todo： 不同的模型，标签的数量不相同
        data = rows[1:-1]       # todo
        random_num = rows[-1]

        for i in range(len(label)):
            label[i] = int(float(label[i]))  # 将label转为0/1整数

        train_lable.append(label)
        train_data.append(data)

    logger.info("get train data and labels success")
    train_data, train_lable = np.array(train_data), np.array(train_lable)
    logger.info("train data shape: %s, train lable shape: %s" % (train_data.shape, train_lable.shape))

    logger.info("train lightgbm model start, please wait...")
    model = LGBMClassifier(n_estimators=100, seed=args.seed)
    model.fit(train_data, train_lable)
    logger.info("train lightgbm model end, save model to: %s", lgb_model_path)
    joblib.dump(model, lgb_model_path)     # todo: 需要根据使用的模型不同，设置不同的模型保存路径
    logger.info("save lightgbm model success")
    logger.info("train lgb model success!")


if __name__ == "__main__":
    logger.info("train model")
    train_lgb_model()

