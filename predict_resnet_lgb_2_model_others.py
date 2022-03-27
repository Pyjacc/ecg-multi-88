# _*_ coding:utf-8 _*_

'''对结果中既有normal又有others的进行校正'''


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
from models.resnet.resnet34_rdrop import resnet341d7_15, resnet342d7_15     # todo

tools.seed_torch(args.seed)

def perdict_with_lgb(device, network1, network2, lgb_model, data_loader, test_label_csv, answer_path):
    output_list = []

    for i, (data, label) in enumerate(tqdm(data_loader)):
        data = data.to(device)
        output1 = network1(data)
        output2 = network2(data)
        output1 = output1.data.cpu().numpy()
        output2 = output2.data.cpu().numpy()
        output1 = output1.tolist()  # 将numpy.ndarray转为二维list
        output2 = output2.tolist()

        output = []
        for i in range(len(output1)):  # 取出二维list的每一行进行拼接，每一行为一张心电图的特征
            out1 = output1[i]
            out2 = output2[i]
            # out = np.array(out1 + out2).transpose()
            out = np.array(out1 + out2)
            output.append(out)

        predict = lgb_model.predict(np.array(output))
        output_list += predict

    # write predict result to answer.csv
    logger.info("writ answer to csv file")
    patient_id_list = []
    test_df = pd.read_csv(test_label_csv)
    for _, rows in test_df.iterrows():
        patient_id = rows[0]
        patient_id_list.append(patient_id)

    df_answer = pd.DataFrame()
    df_answer['patient_id'] = patient_id_list
    df_answer[args.ecg_classes] = output_list
    df_answer.to_csv(answer_path, index=None)


def predict_with_resnet_lgb():
    answer_path = args.answer_lgb_2_model            # todo
    model_1d_path = args.model_path_others_1d
    model_2d_path = args.model_path_others_2d
    lgb_model_path = args.lgb_others_2_model_path
    test_label_csv = args.test_label_csv
    data_dir = os.path.normpath(args.train_data_dir)
    device = tools.get_device(args)
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
        logger.info("test label csv dir: %s" % test_label_csv)
        df.to_csv(test_label_csv, index=None)
        logger.info("create test_label_csv success")

    # 2. 根据测试数据集中的数据进行预测
    logger.info('load test data')
    test_dataset = ECGDataset('test', data_dir, test_label_csv, None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)       # 因为要对每张心电图使用heartpy抽取特征，因此batch_size必须为1

    logger.info("load resnet model")
    model1d = torch.load(model_1d_path, map_location=device)        # todo
    model2d = torch.load(model_2d_path, map_location=device)
    num_class = len(args.ecg_classes)
    num_leads = args.num_leads
    network1 = resnet341d7_15(input_channels=num_leads, num_classes=num_class).to(device)
    network2 = resnet342d7_15(input_channels=num_leads, num_classes=num_class).to(device)
    network1.load_state_dict(model1d, False)
    network2.load_state_dict(model2d, False)
    network1.eval()
    network2.eval()
    logger.info("load resnet model success")

    logger.info('load lightgbm model')
    lgb_model = joblib.load(lgb_model_path)        # todo：根据使用的模型不同，选择相应的模型路径
    logger.info('predict on test data, please wait...')

    args.get_feature = True
    perdict_with_lgb(device, network1, network2, lgb_model, test_loader, test_label_csv, answer_path)
    args.get_feature = False
    logger.info('predict success!')



if __name__ == "__main__":
    predict_with_resnet_lgb()
