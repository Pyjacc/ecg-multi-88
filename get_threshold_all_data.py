# _*_ coding:utf-8 _*_

'''使用所有数据获取阈值'''

import warnings
import numpy as np
import pandas as pd
import joblib
from utils.tools import tools
from utils.tools import logger
from utils.configures import args

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

    logger.info("write optimal thresholds to file")
    with open(threshold_file_path, "w", encoding="utf-8") as fw:        # todo: 需要根据使用的模型不同，将阈值写入相应的文件
        for threshold in thresholds:
            fw.write(str(threshold))
            fw.write("\n")
        fw.close()

    logger.info("thresholds: %s" % thresholds)


def get_feature():
    logger.info("get feature")
    args.get_feature = True
    # GetFeature().get_feature_from_resnet(resnet341d)        # 使用resnet模型抽取所有训练集的特征，并保存到csv文件
    args.get_feature = False


def get_all_data_thresholds():
    feature_path = args.feature_renet_4_model_csv
    lgb_model_path = args.lgb_4_model_path
    threshold_file_path = args.threshold_4_model_lgb

    get_feature()

    df = pd.read_csv(feature_path, sep='\t')          # feature文件路径要根据使用的不同的feature对应的文件而定
    dev_data, dev_label = [], []

    logger.info("get train data and labels start")
    for _, rows in df.iterrows():
        rows = rows[0].split(",")
        label = rows[:12]
        data = rows[12:-1]

        for i in range(len(label)):
            label[i] = int(float(label[i]))  # 将label转为0/1整数

        dev_label.append(label)
        dev_data.append(data)

    logger.info("get train data and labels success")
    dev_data = np.array(dev_data)
    dev_label = np.array(dev_label)
    logger.info("dev data shape: %s, dev lable shape: %s" % (dev_data.shape, dev_label.shape))

    # get optimal thresholds
    model = joblib.load(lgb_model_path)            # 使用已经训练好的lgb模型寻找阈值
    get_threshold(model, dev_data, dev_label, threshold_file_path)
    logger.info("train lgb model success!")


if __name__ == "__main__":
    logger.info("use all data to get thresholds")
    get_all_data_thresholds()

