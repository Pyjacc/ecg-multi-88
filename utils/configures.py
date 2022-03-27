# -*- coding: utf-8 -*-
# @Time    : 2021/11/17 14:10
# @Author  : Qinglong
# @File    : configures.py.py
# @Description: In User Settings Edit

import argparse

plat_form = "local"
# plat_form = "aiwin"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', type=str, default=plat_form, help="train platform, local or aiwin")

    if plat_form == "local":
        # 本地算力资源运行时的文件路径
        parser.add_argument('--train_data_dir', type=str, default='./datasets/heart/task2/Train', help='train data dir')
        parser.add_argument('--trainreference', type=str, default='./datasets/heart/task2/trainreference.csv', help='trainreference.csv dir')
        parser.add_argument('--answer', type=str, default='./outputs/results/answer/answer.csv', help='dir for answer')
    else:
        # 在AIWIN训练平台上运行时的文件路径
        parser.add_argument('--train_data_dir', type=str, default='../../../datasets/heart/task2/Train', help='train data dir')
        parser.add_argument('--trainreference', type=str, default='../../../datasets/heart/task2/trainreference.csv', help='trainreference.csv dir')
        parser.add_argument('--answer', type=str, default='./answer.csv', help='dir for answer')

    parser.add_argument('--trainreference_new', type=str, default='./outputs/results/labels/trainreference_new.csv')
    parser.add_argument('--trainreference_others', type=str, default='./outputs/results/labels/trainreference_others.csv')
    parser.add_argument('--trainreference_normal', type=str, default='./outputs/results/labels/trainreference_normal.csv')
    parser.add_argument('--test_data_txt', type=str, default='./outputs/results/labels/testdata.txt', help='test data file')
    parser.add_argument('--test_data_other_txt', type=str, default='./outputs/results/labels/test_data_other.txt', help='test data file')
    parser.add_argument('--test_data_normal_txt', type=str, default='./outputs/results/labels/test_data_normal.txt',
                        help='test data file')
    parser.add_argument('--dev_data_txt', type=str, default='./outputs/results/labels/devdata.txt', help='dev data file')
    parser.add_argument('--dev_data_other_txt', type=str, default='./outputs/results/labels/dev_data_other.txt', help='dev data file')
    parser.add_argument('--dev_data_normal_txt', type=str, default='./outputs/results/labels/dev_data_normal.txt',
                        help='dev data file')
    parser.add_argument('--mini_sample_txt', type=str, default='./outputs/results/labels/mini_sample.txt',
                        help='mini sample data file')
    parser.add_argument('--mini_sample_other_txt', type=str, default='./outputs/results/labels/mini_sample_other.txt',
                        help='mini sample data file')
    parser.add_argument('--mini_sample_normal_txt', type=str, default='./outputs/results/labels/mini_sample_normal.txt',
                        help='mini sample data file')
    parser.add_argument('--disease_groups', type=str, default='./outputs/results/answer/disease_groups.csv',
                        help='possible combinations of diseases')

    parser.add_argument('--answer_lgb_1d', type=str, default='./outputs/results/answer/answer_lgb_1d.csv')
    parser.add_argument('--answer_lgb_2d', type=str, default='./outputs/results/answer/answer_lgb_2d.csv')
    parser.add_argument('--answer_lgb_4_model', type=str, default='./outputs/results/answer/answer_lgb_4_model.csv')
    parser.add_argument('--answer_lgb_2_model', type=str, default='./outputs/results/answer/answer_lgb_2_model.csv')
    parser.add_argument('--answer_lgb_normal', type=str, default='./outputs/results/answer/answer_lgb_normal.csv')
    parser.add_argument('--answer_lgb_others', type=str, default='./outputs/results/answer/answer_lgb_others.csv')
    parser.add_argument('--answer_xgb', type=str, default='./outputs/results/answer/answer_xgb.csv')
    parser.add_argument('--answer_catb', type=str, default='./outputs/results/answer/answer_catb.csv')
    parser.add_argument('--answer_lr', type=str, default='./outputs/results/answer/answer_lr.csv')
    parser.add_argument('--answer_rf', type=str, default='./outputs/results/answer/answer_rf.csv')
    parser.add_argument('--answer_mlp', type=str, default='./outputs/results/answer/answer_mlp.csv')

    parser.add_argument('--train_label_csv', type=str, default="./outputs/results/labels/train_labels.csv", help="train data label")
    parser.add_argument('--train_label_normal_csv', type=str, default="./outputs/results/steps/train_labels_normal.csv")
    parser.add_argument('--train_label_others_csv', type=str, default="./outputs/results/steps/train_labels_others.csv")
    parser.add_argument('--dev_label_csv', type=str, default="./outputs/results/labels/dev_labels.csv", help="dev data label")
    parser.add_argument('--dev_label_others_csv', type=str, default="./outputs/results/steps/dev_label_others.csv",
                        help="dev data label")
    parser.add_argument('--dev_label_normal_csv', type=str, default="./outputs/results/steps/dev_label_normal.csv",
                        help="dev data label")
    parser.add_argument('--train_dev_label_csv', type=str, default="./outputs/results/labels/train_dev_label.csv", help="dev data label")
    parser.add_argument('--train_dev_label_others_csv', type=str, default="./outputs/results/labels/train_dev_label_others.csv", help="dev data label")
    parser.add_argument('--train_dev_label_normal_csv', type=str, default="./outputs/results/labels/train_dev_label_normal.csv", help="dev data label")
    parser.add_argument('--test_label_csv', type=str, default="./outputs/results/labels/test_labels.csv", help="test data label")
    parser.add_argument('--test_label_normal_csv', type=str, default="./outputs/results/steps/test_label_normal.csv")
    parser.add_argument('--test_label_others_csv', type=str, default="./outputs/results/steps/test_labels_others.csv")
    parser.add_argument('--tmp_test_label_others_csv', type=str, default="./outputs/results/steps/tmp_test_label_others.csv")
    parser.add_argument('--tmp_test_label_normal_csv', type=str, default="./outputs/results/steps/tmp_test_label_normal.csv")

    parser.add_argument('--png_dir', type=str, default='./outputs/results', help='pgn dir')
    parser.add_argument('--log_dir', type=str, default='./outputs/logs', help='log dir')
    parser.add_argument('--feature_train_csv', type=str, default='./outputs/results/feature/feature_train.csv')
    parser.add_argument('--feature_renet_csv', type=str, default='./outputs/results/feature/feature_resnet.csv')
    parser.add_argument('--feature_renet_1d_csv', type=str, default='./outputs/results/feature/feature_renet_1d.csv')
    parser.add_argument('--feature_renet_2d_csv', type=str, default='./outputs/results/feature/feature_renet_2d.csv')
    parser.add_argument('--feature_renet_4_model_csv', type=str, default='./outputs/results/feature/feature_renet_4_model.csv')
    parser.add_argument('--feature_renet_2_model_csv', type=str, default='./outputs/results/feature/feature_renet_2_model.csv')
    parser.add_argument('--feature_renet_normal_csv', type=str, default='./outputs/results/feature/feature_resnet_normal.csv')
    parser.add_argument('--feature_renet_others_csv', type=str, default='./outputs/results/feature/feature_resnet_others.csv')
    parser.add_argument('--feature_heartpy_csv', type=str, default='./outputs/results/feature/feature_heartpy.csv')

    parser.add_argument('--lgb_threshold_data_txt', type=str, default='./outputs/results/labels/lgb_threshold_data.txt')
    parser.add_argument('--threshold_file_lgb_1d', type=str, default='./outputs/results/feature/threshold_lgb_1d.txt')
    parser.add_argument('--threshold_file_lgb_2d', type=str, default='./outputs/results/feature/threshold_lgb_2d.txt')
    parser.add_argument('--threshold_4_model_lgb', type=str, default='./outputs/results/feature/threshold_4_model_lgb.txt')
    parser.add_argument('--threshold_2_model_lgb', type=str, default='./outputs/results/feature/threshold_2_model_lgb.txt')
    parser.add_argument('--threshold_lgb_normal', type=str, default='./outputs/results/feature/threshold_lgb_normal.txt')
    parser.add_argument('--threshold_lgb_others', type=str, default='./outputs/results/feature/threshold_lgb_others.txt')
    parser.add_argument('--threshold_file_xgb', type=str, default='./outputs/results/feature/threshold_xgb.txt')
    parser.add_argument('--threshold_file_catb', type=str, default='./outputs/results/feature/threshold_catb.txt')
    parser.add_argument('--threshold_file_lr', type=str, default='./outputs/results/feature/threshold_lr.txt')
    parser.add_argument('--threshold_file_rf', type=str, default='./outputs/results/feature/threshold_rf.txt')
    parser.add_argument('--threshold_file_mlp', type=str, default='./outputs/results/feature/threshold_mlp.txt')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--seed', type=int, default=1210, help='seed to split data')
    parser.add_argument('--phase', type=str, default='train', help='phase: train or test')
    parser.add_argument('--use_gpu', default=True, action='store_true', help='Use GPU')

    parser.add_argument('--model_type', type=str, default='usual', help='type of different model')
    parser.add_argument('--model_path', type=str, default='./models/ckpt/multilabel_cls.pth', help='path to saved model')
    parser.add_argument('--model_path_1d', type=str, default='./models/ckpt/multilabel_cls_1d.pth', help='path to saved model')
    parser.add_argument('--model_path_1d_3_5', type=str, default='./models/ckpt/multilabel_cls_1d35.pth')
    parser.add_argument('--model_path_1d_7_15', type=str, default='./models/ckpt/multilabel_cls_1d715.pth')
    parser.add_argument('--model_path_2d', type=str, default='./models/ckpt/multilabel_cls_2d.pth', help='path to saved model')
    parser.add_argument('--model_path_2d_3_5', type=str, default='./models/ckpt/multilabel_cls_2d35.pth', help='path to saved model')
    parser.add_argument('--model_path_2d_7_15', type=str, default='./models/ckpt/multilabel_cls_2d715.pth', help='path to saved model')
    parser.add_argument('--model_path_normal_1d', type=str, default='./models/ckpt/normal_cls_1d.pth', help='path to saved normal model')
    parser.add_argument('--model_path_normal_2d', type=str, default='./models/ckpt/normal_cls_2d.pth', help='path to saved normal model')
    parser.add_argument('--model_path_others_1d', type=str, default='./models/ckpt/others_cls_1d.pth', help='path to saved others model')
    parser.add_argument('--model_path_others_2d', type=str, default='./models/ckpt/others_cls_2d.pth',
                        help='path to saved others model')
    parser.add_argument('--lgb_model_path_1d', type=str, default='./models/ckpt/lgb_cls_1d.pkl', help='lightgbm model path')
    parser.add_argument('--lgb_model_path_2d', type=str, default='./models/ckpt/lgb_cls_2d.pkl',
                        help='lightgbm model path')
    parser.add_argument('--lgb_4_model_path', type=str, default='./models/ckpt/lgb_4_model_path.pkl', help='lightgbm model path')
    parser.add_argument('--lgb_2_model_path', type=str, default='./models/ckpt/lgb_2_model_path.pkl', help='lightgbm model path')
    parser.add_argument('--lgb_normal_path', type=str, default='./models/ckpt/lgb_normal_cls.pkl', help='lightgbm model path')
    parser.add_argument('--lgb_others_1d_path', type=str, default='./models/ckpt/lgb_others_1d.pkl')
    parser.add_argument('--lgb_others_2_model_path', type=str, default='./models/ckpt/lgb_others_2_model_path.pkl')
    parser.add_argument('--lgb_normal_2_model_path', type=str, default='./models/ckpt/lgb_normal_2_model_path.pkl')
    parser.add_argument('--xgb_model_path', type=str, default='./models/ckpt/xgb_cls.pkl', help='xgboost model path')
    parser.add_argument('--catb_model_path', type=str, default='./models/ckpt/catb_cls.pkl', help='catboost model path')
    parser.add_argument('--lr_model_path', type=str, default='./models/ckpt/lr_cls.pkl', help='logistic regression model path')
    parser.add_argument('--rf_model_path', type=str, default='./models/ckpt/rf_cls.pkl', help='random forest classifier model path')
    parser.add_argument('--mlp_model_path', type=str, default='./models/ckpt/mlp_cls.pkl', help='mlp model path')

    # 正常心电图、窦性心动过缓、窦性心动过速、窦性心律不齐、心房颤动、室性早搏、房性早搏、一度房室传导阻滞、完全性右束支传导阻滞、
    # T波改变、ST改变、其他
    parser.add_argument('--ecg_classes', type=list, default=['Normal', 'SB', 'NT', 'SI', 'AF', 'PVC', 'PAC', 'IAVB', 'RBBB', 'TA', 'STC', 'Others'],
                        help="classes of ecg")
    parser.add_argument('--ecg_classes_normal', type=list, default=['Normal'], help="normal or normal classes")
    parser.add_argument('--ecg_classes_others', type=list, default=['Others'], help="others or unothers classes")
    parser.add_argument('--ecg_leads', type=list, default=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
                        help='ecg leads')
    parser.add_argument('--num_leads', type=int, default=12, help='ecg leads and add feature leads')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers to load data')
    parser.add_argument('--get_feature', type=bool, default=False, help='switch get feature or not')
    return parser.parse_args()


args = parse_args()