# _*_ coding:utf-8 _*_


'''使用resnet34抽取的feature训练xgboost模型'''

import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
import joblib
from utils.tools import tools
from utils.tools import logger
from utils.configures import args
from utils.features import GetFeature


warnings.filterwarnings('ignore', category=FutureWarning)
tools.seed_torch(args.seed)

# get optimal thresholds
def get_threshold(model, data, label):
    logger.info("get optimal thresholds, please wait...")
    thresholds = []
    y_val_scores = model.predict_proba(data)
    for i in range(len(args.ecg_classes)):
        y_val_score = y_val_scores[:, i]
        threshold = tools.find_optimal_threshold(label[:, i], y_val_score)
        thresholds.append(threshold)

    logger.info("write optimal thresholds to file")
    with open(args.threshold_file_xgb, "w", encoding="utf-8") as fw:        # todo: 需要根据使用的模型不同，将阈值写入相应的文件
        for threshold in thresholds:
            fw.write(str(threshold))
            fw.write("\n")
        fw.close()

    logger.info("thresholds: %s" % thresholds)


def get_feature():
    logger.info("get feature")
    args.get_feature = True
    # GetFeature().get_feature_from_resnet()        # 使用resnet模型抽取所有训练集的特征，并保存到csv文件
    args.get_feature = False


def train_xgb_model():
    get_feature()

    train_random_num, dev_random_num = tools.split_train_dev_data(seed=args.seed)
    df = pd.read_csv(args.feature_renet_csv, sep='\t')          # feature文件路径要根据使用的不同的feature对应的文件而定
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

    logger.info("train xgb model start, please wait...")
    model = XGBClassifier(n_estimators=100, tree_method='gpu_hist', gpu_id=0, seed=args.seed)       # todo: 无gpu时将tree_method改为auto
    model = OneVsRestClassifier(model)      # sklearn提供的multiclass子类OneVsRestClassifier实现多标签分类
    model.fit(train_data, train_lable)
    logger.info("train xgb model end, save model")
    joblib.dump(model, args.xgb_model_path)        # todo: 需要根据使用的模型不同，设置不同的模型保存路径
    logger.info("save xgb model success")

    # get optimal thresholds
    model = joblib.load(args.xgb_model_path)       # todo: 需要根据使用的模型不同，设置不同的模型保存路径
    get_threshold(model, dev_data, dev_label)
    logger.info("train xgb model success!")


if __name__ == "__main__":
    logger.info("train model")
    train_xgb_model()

