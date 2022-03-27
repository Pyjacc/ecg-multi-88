# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 14:06
# @Author  : Qinglong
# @File    : predict_resnet.py.py
# @Description: In User Settings Edit

'''仅仅使用resnet模型进行预测'''


import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils.tools import tools
from utils.tools import logger
from utils.configures import args
from utils.dataset import ECGDataset
from models.resnet import resnet34

tools.seed_torch(args.seed)

def predict(data_loader, net, device, thresholds):
    output_list = []

    for _, (data, label) in enumerate(tqdm(data_loader)):
        data = data.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())

    y_hats = np.vstack(output_list)     # y_hats.shape:(2800, 12)
    y_preds = []

    for i in range(len(args.ecg_classes)):
        y_hat = y_hats[:, i]
        # y_pred = (y_hat >= 0.5).astype(int)
        y_pred = (y_hat >= thresholds[i]).astype(int)
        y_preds.append(y_pred)

    y_preds = np.array(y_preds).transpose()

    # write predict result to answer.csv
    patient_id_list = []
    test_df = pd.read_csv(args.test_label_csv)
    for _, rows in test_df.iterrows():
        patient_id = rows[0]
        patient_id_list.append(patient_id)

    df_answer = pd.DataFrame()
    df_answer['patient_id'] = patient_id_list
    df_answer[args.ecg_classes] = y_preds
    df_answer.to_csv(args.answer, index=None)


def model_predict():
    device = tools.get_device(args)
    logger.info('device: %s' % device)
    data_dir = os.path.normpath(args.test_data_dir)
    logger.info("test data dir: %s" % data_dir)

    # 1. 使用aiwin平台进行预测时，先生成test_label_csv
    if args.platform != "local":
        logger.info("create test_label_csv start")

        patient_id_list = []
        mat_file = os.listdir(data_dir)
        logger.info("get all mat file")
        for file in mat_file:
            if file.endswith(".mat"):
                # logger.info("get one mat file: %s" % file)
                patient_id = file.split(".")[0]
                patient_id_list.append(patient_id)

        logger.info("create test label csv, test data len: %d" % len(patient_id_list))
        df = pd.DataFrame(data=patient_id_list, columns=["patient_id"])
        logger.info("test label csv dir: %s" % args.test_label_csv)
        df.to_csv(args.test_label_csv, index=None)
        logger.info("create test_label_csv success")

    # 2. 根据测试数据集中的数据进行预测
    # args.batch_size = 1
    logger.info('load test data')
    test_dataset = ECGDataset('test', data_dir, args.test_label_csv, None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    logger.info("load model")
    model = torch.load(args.model_path, map_location=device)
    logger.info("load resnet network")
    num_class = len(args.ecg_classes)
    num_leads = args.num_leads
    network = resnet34(input_channels=num_leads, num_classes=num_class).to(device)
    logger.info("load state dict")
    network.load_state_dict(model, False)
    logger.info("load model success")
    network.eval()

    logger.info('results on test data:')
    thresholds = [0.5757, 0.5656, 0.4848, 0.5151, 0.6363, 0.4444, 0.5555, 0.8080, 0.4343, 0.3232, 0.1515, 0.2929]
    predict(test_loader, network, device, thresholds)
    logger.info('test end!')


def parse_answer():
    result = []
    result_df = pd.read_csv(args.answer, sep='\t')

    for _, rows in result_df.iterrows():
        rows = rows[0].split(",")
        patient_id = rows[0]

        res = str(patient_id)
        for idx, disease in enumerate(rows[1:]):
            if disease == "1":
                res = res + "," + str(idx + 1)

        result.append(res)

    with open("./result_answer.csv", "w", encoding="utf-8") as fw:
        for i in result:
            fw.write(str(i))
            fw.write("\n")
        fw.close()



if __name__ == "__main__":
    logger.info("predict start...")
    # model_predict()
    parse_answer()          # 对预测出的answer.csv进行转换，转换为aiwin需要的格式