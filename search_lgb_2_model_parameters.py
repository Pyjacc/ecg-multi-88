# _*_ coding:utf-8 _*_


'''使用resnet34_1d和resnet34_2d共计2个模型抽取的feature训练lightgbm模型'''

import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from utils.tools import tools
from utils.tools import logger
from utils.configures import args
from utils.features import GetFeature
from models.resnet.resnet34_rdrop import resnet341d7_15, resnet342d7_15     # todo


warnings.filterwarnings('ignore', category=FutureWarning)


def get_feature(model_1d_path1, model_2d_path1, feature_path):
    logger.info("get feature")
    args.get_feature = True
    # 使用resnet模型抽取所有训练集的特征，并保存到csv文件
    GetFeature().get_feature_from_2_resnet(model_1d_path1, model_2d_path1, resnet341d7_15, resnet342d7_15,
                                           feature_path)
    args.get_feature = False


def train_lgb_model():
    model_1d_path_7_15 = args.model_path_1d_7_15          # todo: 需要修改的变量
    model_2d_path_7_15 = args.model_path_2d_7_15
    feature_path = args.feature_renet_2_model_csv

    get_feature(model_1d_path_7_15, model_2d_path_7_15, feature_path)

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

    search_optimal_parameters(train_data, train_lable)


def search_optimal_parameters(train_data, train_lable):
    logger.info("search optimal parameters start!")
    model = LGBMClassifier(
        random_state=args.seed,     # 随机种子
        is_unbalance=True,
        n_estimators=100,           # 拟合的树的棵树（弱学习器个数）
        learning_rate=0.1,          # 学习率
        max_depth=-1,               # 最大树的深度。每个弱学习器也就是决策树的最大深度。-1表示不限制。
        num_leaves=31,              # 树的最大叶子数
        device='cpu'
    )

    model = OneVsRestClassifier(model)

    # 第一步： 调整max_depth和num_leaves
    params = {'estimator__max_depth': [4, 6, 8, 10, 12],
              'estimator__num_leaves': [4, 6, 8, 10, 12],
              'estimator__n_estimators': [50, 60, 70, 80, 100, 120],
              'estimator__learning_rate': [0.001, 0.005, 0.1, 0.5],
              'estimator__is_unbalance': [True]}
    model_search = GridSearchCV(model, params, cv=5, scoring="roc_auc", n_jobs=-1)
    model_search.fit(train_data, train_lable)
    logger.info("search result:")
    logger.info(model_search.best_params_)
    print("------------------------------------")
    logger.info(model_search.best_score_)
    print("=====================================")
    # logger.info(model_search.cv_results_['mean_test_score'])
    # logger.info(model_search.cv_results_['params'])
    logger.info("search optimal parameters success!")

if __name__ == "__main__":
    logger.info("train model")
    train_lgb_model()

