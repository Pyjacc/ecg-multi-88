# _*_ coding:utf-8 _*_


'''使用resnet341d7_15和resnet342d7_15两个模型抽取的feature训练lightgbm模型'''

import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from utils.tools import logger
from utils.configures import args
from steps.other.feature_other import GetFeature      # todo,不同的模型对应不同的GetFeature
from models.resnet.resnet34_rdrop import resnet341d7_15, resnet342d7_15     # todo


warnings.filterwarnings('ignore', category=FutureWarning)


def get_feature(model_1d_path, model_2d_path, feature_path):
    logger.info("get feature")
    args.get_feature = True
    # 使用resnet模型抽取所有训练集的特征，并保存到csv文件
    GetFeature().get_feature_from_2_resnet(model_1d_path, model_2d_path, resnet341d7_15, resnet342d7_15, feature_path)
    args.get_feature = False


def train_lgb_model():
    model_1d_path = args.model_path_others_1d          # todo: 需要修改的变量
    model_2d_path = args.model_path_others_2d
    feature_path = args.feature_renet_others_csv

    get_feature(model_1d_path,  model_2d_path, feature_path)

    # 划分训练数据和寻找阈值的数据
    df = pd.read_csv(feature_path, sep='\t')          # feature文件路径要根据使用的不同的feature对应的文件而定
    train_data, train_lable = [], []

    logger.info("get train data and labels start")
    for _, rows in df.iterrows():
        rows = rows[0].split(",")
        label = rows[:1]
        data = rows[1:-1]
        random_num = rows[-1]

        for i in range(len(label)):
            label[i] = int(float(label[i]))  # 将label转为0/1整数

        train_lable.append(label)
        train_data.append(data)


    logger.info("get train data and labels success")
    train_data, train_lable = np.array(train_data), np.array(train_lable)
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
    params = {'estimator__max_depth': [2, 3, 4],
              'estimator__num_leaves': [4, 5, 6],
              'estimator__n_estimators': [40, 50, 60],
              'estimator__learning_rate': [0.0001, 0.001, 0.005, 0.1, 0.5],
              'estimator__is_unbalance': [True]}
    model_search = GridSearchCV(model, params, cv=3, scoring="roc_auc", n_jobs=-1, verbose=2)
    model_search.fit(train_data, train_lable)
    logger.info("search result:")
    logger.info(model_search.best_params_)
    logger.info(model_search.best_score_)
    # logger.info(model_search.cv_results_['mean_test_score'])
    # logger.info(model_search.cv_results_['params'])
    logger.info("search optimal parameters success!")

if __name__ == "__main__":
    logger.info("train model")
    train_lgb_model()

