# _*_ coding:utf-8 _*_

'''先使用resnet34提取特征，然后将特征送入lr进行多分类预测结果'''


import os
import torch
import numpy as np
from tqdm import tqdm
import joblib
import pandas as pd
from torch.utils.data import DataLoader
from utils.tools import tools
from utils.tools import logger
from utils.configures import args
from utils.dataset import ECGDataset
from models.resnet import resnet34

tools.seed_torch(args.seed)

def perdict_with_lr(device, network, model, data_loader, thresholds):
    output_list = []

    for i, (data, label) in enumerate(tqdm(data_loader)):
        data = data.to(device)
        output = network(data)
        output = output.data.cpu().numpy()
        predict = model.predict_proba(output)
        output_list.append(predict)

    y_hats = np.vstack(output_list)     # y_hats.shape:(2800, 12)
    y_preds = []

    for i in range(len(args.ecg_classes)):
        y_hat = y_hats[:, i]
        # y_pred = (y_hat >= 0.5).astype(int)
        y_pred = (y_hat >= thresholds[i]).astype(int)
        y_preds.append(y_pred)

    y_preds = np.array(y_preds).transpose()

    # write predict result to answer.csv
    logger.info("writ answer to csv file")
    patient_id_list = []
    test_df = pd.read_csv(args.test_label_csv)
    for _, rows in test_df.iterrows():
        patient_id = rows[0]
        patient_id_list.append(patient_id)

    df_answer = pd.DataFrame()
    df_answer['patient_id'] = patient_id_list
    df_answer[args.ecg_classes] = y_preds
    df_answer.to_csv(args.answer_lr, index=None)


def predict_with_resnet_lr():
    device = tools.get_device(args)
    data_dir = os.path.normpath(args.test_data_dir)
    logger.info("device: %s, test data dir: %s" % (device, data_dir))

    # 1. 使用aiwin平台进行预测时，先生成test_label_csv
    if args.platform != "local":
        logger.info("create test_label_csv start")

        patient_id_list = []
        mat_file = os.listdir(data_dir)
        logger.info("get all mat file")
        for file in mat_file:
            if file.endswith(".mat"):
                patient_id = file.split(".")[0]
                patient_id_list.append(patient_id)

        logger.info("create test label csv, test data len: %d" % len(patient_id_list))
        df = pd.DataFrame(data=patient_id_list, columns=["patient_id"])
        logger.info("test label csv dir: %s" % args.test_label_csv)
        df.to_csv(args.test_label_csv, index=None)
        logger.info("create test_label_csv success")

    # 2. 根据测试数据集中的数据进行预测
    logger.info('load test data')
    test_dataset = ECGDataset('test', data_dir, args.test_label_csv, None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)       # 因为要对每张心电图使用heartpy抽取特征，因此batch_size必须为1

    logger.info("load resnet model")
    model = torch.load(args.model_path, map_location=device)
    logger.info("load resnet network")
    num_class = len(args.ecg_classes)
    num_leads = args.num_leads
    network = resnet34(input_channels=num_leads, num_classes=num_class).to(device)
    logger.info("load state dict")
    network.load_state_dict(model)
    network.eval()
    logger.info("load resnet model success")

    logger.info('load lightgbm model')
    lgb_model = joblib.load(args.lgb_model_path)
    logger.info('predict on test data, please wait...')

    # get thresholds
    thresholds = []
    if os.path.exists(args.threshold_file_lr):
        with open(args.threshold_file_lr, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            for line in lines:
                threshold = float(line)
                thresholds.append(threshold)
        logger.info("thresholds: %s" % thresholds)
    else:
        logger.info("thresholds file is not exists")
        thresholds = [0.2323, 0.0606, 0.1010, 0.2020, 0.3636, 0.2929, 0.0202, 0.2323, 0.9797, 0.4343, 0.3939, 0.2222]

    args.get_feature = True
    perdict_with_lr(device, network, lgb_model, test_loader, thresholds)
    args.get_feature = False
    logger.info('predict success!')



if __name__ == "__main__":
    predict_with_resnet_lr()
