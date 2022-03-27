# _*_ coding:utf-8 _*_


'''使用resnet34抽取的feature训练lightgbm模型'''

import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from utils.tools import tools
from utils.tools import logger
from utils.configures import args
from utils.features import GetFeature
from models.resnet.resnet34_rdrop import resnet341d7_15


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


def get_feature(resnet_model_path, feature_path):
    logger.info("get feature")
    args.get_feature = True
    GetFeature().get_feature_from_resnet(resnet341d7_15, resnet_model_path, feature_path)        # 使用resnet模型抽取所有训练集的特征，并保存到csv文件
    args.get_feature = False


def train_lgb_model():
    resnet_model_path = args.model_path_1d
    feature_path = args.feature_renet_1d_csv
    lgb_model_path = args.lgb_model_path_1d
    threshold_file_path = args.threshold_file_lgb_1d

    get_feature(resnet_model_path, feature_path)

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

def search_optimal_parameters(train_data, train_lable):
    logger.info("search optimal parameters start!")
    model = LGBMClassifier(
        random_state=args.seed,     # 随机种子
        is_unbalance=True,
        n_estimators=100,           # 拟合的树的棵树（弱学习器个数）
        learning_rate=0.1,          # 学习率
        max_depth=-1,               # 最大树的深度。每个弱学习器也就是决策树的最大深度。-1表示不限制。
        num_leaves=31,              # 树的最大叶子数
        feature_fraction=0.9,
        min_child_samples=3,
        min_child_weight=0.001,     # 分支结点的最小权重
        reg_alpha=0.0001,
        reg_lambda=0.0001,
        cat_smooth=0,
        device='cpu'
    )
    model = OneVsRestClassifier(model)

    # 第一步： 调整max_depth和num_leaves
    params = {'estimator__max_depth': [4, 6, 8, 10, 12, 14, 16],
              'estimator__num_leaves': [5, 10, 15, 20, 25, 30],
              'estimator__n_estimators': [50, 60, 70, 80],
              'estimator__is_unbalance': [True]}
    model_search = GridSearchCV(model, params, cv=5, scoring="f1", n_jobs=-1, verbose=2)
    model_search.fit(train_data, train_lable)
    logger.info("search result:")
    logger.info(model_search.best_params_)
    logger.info(model_search.best_score_)
    logger.info("search optimal parameters success!")


if __name__ == "__main__":
    logger.info("train model")
    train_lgb_model()

