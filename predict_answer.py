# _*_ coding:utf-8 _*_

'''
(1)通过resnet341d7_15和resnet342d7_15抽取特征，然后训练lgb_2_model_path模型。
(2)使用lgb_2_model_path模型预测结果。
(3)使用lgb_2_model_path对结果进行校正
'''

import os
import pandas as pd
import torch
import joblib
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from steps.normal.dataset_normal import ECGDataset  # todo
from utils.configures import args
from utils.tools import tools, logger
from models.resnet.resnet34_rdrop import resnet341d7_15, resnet342d7_15
from predict_resnet_lgb_2_model import predict_with_resnet_lgb


def perdict_with_lgb_other(device, network1, network2, lgb_model, data_loader, test_label_csv, answer_path):
    output_list = []

    for i, (data, label, patientid) in enumerate(tqdm(data_loader)):
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
        output_list += predict.tolist()

    # write predict result to answer.csv
    logger.info("writ answer to csv file")
    patient_id_list = []
    test_df = pd.read_csv(test_label_csv)
    for _, rows in test_df.iterrows():
        patient_id = rows[0]
        patient_id_list.append(patient_id)

    df_answer = pd.DataFrame()
    df_answer['patient_id'] = patient_id_list
    df_answer['Normal'] = output_list       # todo
    df_answer.to_csv(answer_path, index=None)


def predict():
    # 第一步： 使用lgb_2_model_path模型预测结果：对应文件predict_resnet_lgb_2_model.py
    predict_with_resnet_lgb()

    # 第二步： 读取第一步中预测的结果文件，查看哪些patient_id对应的结果中即含有normal又含有其他病种，
    # 将同时含有normal和其他病的patient_id写入临时文件。
    patient_id_list = []
    df1 = pd.read_csv(args.answer_lgb_2_model, sep='\t')  # todo:注意第一步中结果文件的名称
    for _, rows in df1.iterrows():
        rows = rows[0].split(',')
        patient_id = rows[0]
        disease = rows[1:]
        if int(disease[0]) == 1 and len(disease) > 1:
            patient_id_list.append(patient_id)

    # 将需要进行进一步预测的patient_id写入临时的csv文件
    df2 = pd.DataFrame(data=patient_id_list, columns=["patient_id"])
    df2.to_csv(args.tmp_test_label_normal_csv, index=None)  # todo

    # 第三步： 针对需要进一步预测的样本，预测其是normal还是其他病种
    answer_path = args.answer_lgb_normal  # todo:第二步保存结果的文件
    model_1d_path = args.model_path_normal_1d
    model_2d_path = args.model_path_normal_2d
    lgb_model_path = args.lgb_normal_2_model_path
    test_label_csv = args.tmp_test_label_normal_csv  # todo：注意用中间结果文件的路径
    data_dir = os.path.normpath(args.train_data_dir)
    device = tools.get_device(args)
    logger.info("device: %s, test data dir: %s" % (device, data_dir))

    # 根据测试数据集中的数据进行预测
    logger.info('load test data')
    test_dataset = ECGDataset('test', data_dir, test_label_csv, None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    logger.info("load resnet model")
    model1d = torch.load(model_1d_path, map_location=device)  # todo
    model2d = torch.load(model_2d_path, map_location=device)
    num_class = len(args.ecg_classes_normal)  # todo
    num_leads = args.num_leads
    network1 = resnet341d7_15(input_channels=num_leads, num_classes=num_class).to(device)
    network2 = resnet342d7_15(input_channels=num_leads, num_classes=num_class).to(device)
    network1.load_state_dict(model1d, False)
    network2.load_state_dict(model2d, False)
    network1.eval()
    network2.eval()
    logger.info("load resnet model success")

    logger.info('load lightgbm model')
    lgb_model = joblib.load(lgb_model_path)  # todo：根据使用的模型不同，选择相应的模型路径
    logger.info('predict on test data, please wait...')

    args.get_feature = True
    perdict_with_lgb_other(device, network1, network2, lgb_model, test_loader, test_label_csv, answer_path)
    args.get_feature = False
    logger.info('predict success!')

    # 第四步：合并答案，将normal模型预测的结果保存在字典中
    normal_disease = dict()
    df3 = pd.read_csv(answer_path, sep='\t')
    for _, rows in df3.iterrows():
        rows = rows[0].split(',')
        patientid = rows[0]
        disease = rows[1]
        normal_disease[patientid] = disease

    result = []
    df4 = pd.read_csv(args.answer_lgb_2_model, sep='\t')        # todo:第一步预测的结果文件
    for _, rows in df4.iterrows():
        rows = rows[0].split(',')
        patient_id = rows[0]
        disease = rows[1:]
        # 含有normal并且还含有其他疾病
        if int(disease[0]) == 1:
            remain11 = disease[1:]
            sum = 0
            for i in range(len(remain11)):
                sum += int(remain11[i])
            if sum > 0:         # 含有其他病
                if patient_id in normal_disease.keys():
                    if int(normal_disease[patient_id]) == 1:  # 是normal,则将除normal以外的所有病赋值为0
                        rows[2:] = [0] * 11
                    else:       # 不是normal
                        rows[1] = 0
        result.append(rows)

    # 将最终结果写入文件
    df2 = pd.DataFrame(data=result, columns=['patient_id'] + args.ecg_classes)
    df2.to_csv(args.answer, index=None)


if __name__ == "__main__":
    predict()
