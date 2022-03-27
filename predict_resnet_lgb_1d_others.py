# _*_ coding:utf-8 _*_

'''先使用resnet34提取特征，然后将特征送入lightgbm模型，识别是others还是非others'''


import os
import torch
from tqdm import tqdm
import joblib
import pandas as pd
from torch.utils.data import DataLoader
from utils.tools import tools
from utils.tools import logger
from utils.configures import args
from steps.other.dataset_other import ECGDataset        # todo： 根据不同的模型选择ECGDataset
from models.resnet.resnet34_rdrop import resnet341d7_15

tools.seed_torch(args.seed)

def perdict_with_lgb(device, network, model, data_loader, test_label_cav, answer_path):
    output_list = []

    for i, (data, label, patientid) in enumerate(tqdm(data_loader)):
        data = data.to(device)
        output = network(data)
        output = output.data.cpu().numpy()
        predict = model.predict(output)
        output_list += predict.tolist()

   # write predict result to answer.csv
    logger.info("writ answer to csv file")
    patient_id_list = []
    test_df = pd.read_csv(test_label_cav)      # todo
    for _, rows in test_df.iterrows():
        patient_id = rows[0]
        patient_id_list.append(patient_id)

    df_answer = pd.DataFrame()
    df_answer['patient_id'] = patient_id_list
    df_answer['Others'] = output_list                           # todo
    df_answer.to_csv(answer_path, index=None)       # todo：根据使用的模型不同，将answer写入不同的文件中


def predict_with_resnet_lgb():
    answer_path = args.answer_lgb_others
    model_1d_path = args.model_path_others_1d
    lgb_model_path = args.lgb_others_1d_path
    test_label_cav = args.test_label_others_csv
    device = tools.get_device(args)
    data_dir = os.path.normpath(args.train_data_dir)
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
        logger.info("test label normal csv dir: %s" % args.test_label_others_csv)
        df.to_csv(args.test_label_others_csv, index=None)       # todo
        logger.info("create test label normal csv success")

    # 2. 根据测试数据集中的数据进行预测
    logger.info('load test data')
    test_dataset = ECGDataset('test', data_dir, test_label_cav, None)      # todo：根据不同的模型选择测试集路径
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    logger.info("load resnet model：%s", model_1d_path)
    model = torch.load(model_1d_path, map_location=device)        # todo: 根据不同的模型选择模型路径
    logger.info("load resnet network")
    num_class = len(args.ecg_classes_others)       # todo: 根据不同的模型选择
    num_leads = args.num_leads
    network = resnet341d7_15(input_channels=num_leads, num_classes=num_class).to(device)
    logger.info("load state dict")
    network.load_state_dict(model, False)
    network.eval()
    logger.info("load resnet model success")

    logger.info('load lightgbm model')
    lgb_model = joblib.load(lgb_model_path)        # todo：根据使用的模型不同，选择相应的模型路径
    logger.info('predict on test data, please wait...')

    args.get_feature = True
    perdict_with_lgb(device, network, lgb_model, test_loader, test_label_cav, answer_path)
    args.get_feature = False
    logger.info('predict success!')



if __name__ == "__main__":
    predict_with_resnet_lgb()
