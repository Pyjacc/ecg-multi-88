# _*_ coding:utf-8 _*_

'''
（1）使用训练好的resent网络提取心电图的特征.
（2）使用heartpy抽取心电图每个导联的特征。
'''

import os
import torch
import scipy.io as sio
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils.tools import tools
from utils.tools import logger
from utils.configures import args
from utils.dataset import ECGDataset


class GetFeature:
    def __init__(self):
        pass

    def join_train_dev_labels_csv(self):
        result_train = []
        df_train = pd.read_csv(args.train_label_csv, sep='\t')
        length_train = len(df_train)
        for _, rows in df_train.iterrows():
            rows = rows[0].split(',')
            result_train.append(rows[:-1])

        result_dev = []
        df_dev = pd.read_csv(args.dev_label_csv, sep='\t')
        length_dev = len(df_dev)
        for _, rows in df_dev.iterrows():
            rows = rows[0].split(',')
            result_dev.append(rows[:-1])

        result = result_train + result_dev
        length = length_train + length_dev
        random_num = np.zeros(length, dtype=np.int8)
        for i in range(10):
            start = int(length * i / 10)
            end = int(length * (i + 1) / 10)
            random_num[start:end] = i + 1

        df = pd.DataFrame(data=result, columns=['patient_id'] + args.ecg_classes)
        df['random'] = np.random.RandomState(args.seed).permutation(random_num)
        columns = df.columns
        df[columns].to_csv(args.train_dev_label_csv, index=None)

    # 使用已经训练好的resent34模型，提取每张心电图的特征。
    def get_feature_from_resnet(self, renset, resnet_model_path, feature_path):
        '''
        （1）使用训练好的resent网络提取心电图的特征，作为lightgbm模型训练的输入数据.
        （2）将所有心电图的特征数据保存在一个csv文件中.
        （3）一张心电图的特征大小为1*1024.
        '''
        device = tools.get_device(args)
        data_dir = os.path.normpath(args.train_data_dir)
        logger.info("device: %s, test data dir: %s" % (device, data_dir))

        # 将train_label_csv和dev_label_csv合并到一个文件中:train_dev_label_csv
        self.join_train_dev_labels_csv()

        logger.info('load train data')
        train_random_num = tools.get_random_num(seed=args.seed)
        train_dataset = ECGDataset('train', data_dir, args.train_dev_label_csv, train_random_num)       # todo
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

        logger.info("load model, model path: %s", args.model_path_1d)
        model = torch.load(resnet_model_path, map_location=device)      # todo
        logger.info("load resnet network")
        num_class = len(args.ecg_classes)
        num_leads = args.num_leads
        network = renset(input_channels=num_leads, num_classes=num_class).to(device)
        logger.info("load state dict")
        network.load_state_dict(model, False)
        logger.info("load model success")
        network.eval()

        logger.info('get feature from data, please wait...')
        self.get_feature_resnet(train_loader, network, device, feature_path)
        logger.info('get feature success!')

    def get_feature_resnet(self, data_loader, net, device, feature_path):
        ecg_features = []
        length = 0

        for _, (data, label, patientid) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output = net(data)
            output = output.data.cpu().numpy()  # 将torch.Tensor转为numpy.ndarray
            label = label.data.cpu().numpy()
            output = output.tolist()  # 将numpy.ndarray转为二维list
            label = label.tolist()

            for i in range(len(output)):  # 取出二维list的每一行进行拼接，每一行为一张心电图的特征
                out = output[i]
                lab = label[i]
                one_result = np.array(lab + out)
                ecg_features.append(one_result)
                length += 1

        # 随机生成标签，用于标记训练集和验证集
        random_num = np.zeros(length, dtype=np.int8)
        for i in range(10):
            start = int(length * i / 10)
            end = int(length * (i + 1) / 10)
            random_num[start:end] = i + 1

        # write feature to csv
        df = pd.DataFrame(data=ecg_features)
        df['random'] = np.random.RandomState(args.seed).permutation(random_num)
        df.to_csv(feature_path, index=None)         # todo


    # 使用已经训练好的resent34_1d和resent34_2d两个模型，提取每张心电图的特征。
    def get_feature_from_2_resnet(self, model_1d_path, model_2d_path, renset1d, renset2d, feature_path):
        '''
        （1）使用训练好的resent网络提取心电图的特征，作为lightgbm模型训练的输入数据.
        （2）将所有心电图的特征数据保存在一个csv文件中.
        （3）一张心电图的特征大小为1*2096.
        '''
        device = tools.get_device(args)
        data_dir = os.path.normpath(args.train_data_dir)    # todo
        logger.info("device: %s, test data dir: %s" % (device, data_dir))

        # 将train_label_csv和dev_label_csv合并到一个文件中:train_dev_label_csv
        self.join_train_dev_labels_csv()        # todo

        logger.info('load train data')
        train_random_num = tools.get_random_num(seed=args.seed)
        train_dataset = ECGDataset('train', data_dir, args.train_dev_label_csv, train_random_num)       # todo
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

        logger.info("load model, model path: %s, %s" % (model_1d_path, model_2d_path))
        model1d = torch.load(model_1d_path, map_location=device)       # todo
        model2d = torch.load(model_2d_path, map_location=device)       # todo

        num_class = len(args.ecg_classes)
        num_leads = args.num_leads
        network1d = renset1d(input_channels=num_leads, num_classes=num_class).to(device)        # todo
        network2d = renset2d(input_channels=num_leads, num_classes=num_class).to(device)        # todo

        network1d.load_state_dict(model1d, False)
        network2d.load_state_dict(model2d, False)
        network1d.eval()
        network2d.eval()
        logger.info("load model success")

        logger.info('get feature from data, please wait...')
        self.get_feature_2_resnet(train_loader, network1d, network2d, device, feature_path)
        logger.info('get feature success!')

    def get_feature_2_resnet(self, data_loader, net1d, net2d, device, feature_path):
        ecg_features = []
        length = 0

        for _, (data, label, patientid) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output1 = net1d(data)
            output2 = net2d(data)
            output1 = output1.data.cpu().numpy()        # 将torch.Tensor转为numpy.ndarray
            output2 = output2.data.cpu().numpy()
            label = label.data.cpu().numpy()
            output1 = output1.tolist()                  # 将numpy.ndarray转为二维list
            output2 = output2.tolist()
            label = label.tolist()

            for i in range(len(output1)):               # 取出二维list的每一行进行拼接，每一行为一张心电图的特征
                out1 = output1[i]
                out2 = output2[i]
                lab = label[i]
                patient_id = patientid[i]

                one_result = np.array(lab + out1 + out2)
                ecg_features.append(one_result)
                length += 1

        # 随机生成标签，用于标记训练集和验证集
        random_num = np.zeros(length, dtype=np.int8)
        for i in range(10):
            start = int(length * i / 10)
            end = int(length * (i + 1) / 10)
            random_num[start:end] = i + 1

        # write feature to csv
        df = pd.DataFrame(data=ecg_features)
        df['random'] = np.random.RandomState(args.seed).permutation(random_num)
        df.to_csv(feature_path, index=None)


    # 使用已经训练好的resent34_1d和resent34_2d两个模型，提取每张心电图的特征。
    def get_feature_from_4_resnet(self, model_1d_path1, model_1d_path2, model_2d_path1, model_2d_path2,
                                  renset1d1, renset1d2, renset2d1, renset2d2, feature_path):
        '''
        （1）使用训练好的resent网络提取心电图的特征，作为lightgbm模型训练的输入数据.
        （2）将所有心电图的特征数据保存在一个csv文件中.
        （3）一张心电图的特征大小为1*2096.
        '''
        device = tools.get_device(args)
        data_dir = os.path.normpath(args.train_data_dir)    # todo
        logger.info("device: %s, test data dir: %s" % (device, data_dir))

        # 将train_label_csv和dev_label_csv合并到一个文件中:train_dev_label_csv
        self.join_train_dev_labels_csv()

        logger.info('load train data')
        train_random_num = tools.get_random_num(seed=args.seed)
        train_dataset = ECGDataset('train', data_dir, args.train_dev_label_csv, train_random_num)       # todo
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

        logger.info("load model, model path: %s, %s, %s, %s" % (model_1d_path1, model_1d_path2, model_2d_path1, model_2d_path2))
        model1d1 = torch.load(model_1d_path1, map_location=device)       # todo
        model1d2 = torch.load(model_1d_path2, map_location=device)
        model2d1 = torch.load(model_2d_path1, map_location=device)       # todo
        model2d2 = torch.load(model_2d_path2, map_location=device)

        num_class = len(args.ecg_classes)
        num_leads = args.num_leads
        network1d1 = renset1d1(input_channels=num_leads, num_classes=num_class).to(device)        # todo
        network1d2 = renset1d2(input_channels=num_leads, num_classes=num_class).to(device)
        network2d1 = renset2d1(input_channels=num_leads, num_classes=num_class).to(device)        # todo
        network2d2 = renset2d2(input_channels=num_leads, num_classes=num_class).to(device)

        network1d1.load_state_dict(model1d1, False)
        network1d2.load_state_dict(model1d2, False)
        network2d1.load_state_dict(model2d1, False)
        network2d2.load_state_dict(model2d2, False)
        network1d1.eval()
        network1d2.eval()
        network2d1.eval()
        network2d2.eval()
        logger.info("load model success")

        logger.info('get feature from data, please wait...')
        self.get_feature_4_resnet(train_loader, network1d1, network1d2, network2d1, network2d2, device, feature_path)
        logger.info('get feature success!')

    def get_feature_4_resnet(self, data_loader, net1d1, net1d2, net2d1, net2d2, device, feature_path):
        ecg_features = []
        length = 0

        for _, (data, label, patientid) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output1 = net1d1(data)
            output2 = net2d1(data)
            output3 = net1d2(data)
            output4 = net2d2(data)

            output1 = output1.data.cpu().numpy()        # 将torch.Tensor转为numpy.ndarray
            output2 = output2.data.cpu().numpy()
            output3 = output3.data.cpu().numpy()
            output4 = output4.data.cpu().numpy()
            label = label.data.cpu().numpy()

            output1 = output1.tolist()                  # 将numpy.ndarray转为二维list
            output2 = output2.tolist()
            output3 = output3.tolist()
            output4 = output4.tolist()
            label = label.tolist()

            for i in range(len(output1)):               # 取出二维list的每一行进行拼接，每一行为一张心电图的特征
                out1 = output1[i]
                out2 = output2[i]
                out3 = output3[i]
                out4 = output4[i]
                lab = label[i]

                one_result = np.array(lab + out1 + out2 + out3 + out4)
                ecg_features.append(one_result)
                length += 1

        # 随机生成标签，用于标记训练集和验证集
        random_num = np.zeros(length, dtype=np.int8)
        for i in range(10):
            start = int(length * i / 10)
            end = int(length * (i + 1) / 10)
            random_num[start:end] = i + 1

        # write feature to csv
        df = pd.DataFrame(data=ecg_features)
        df['random'] = np.random.RandomState(args.seed).permutation(random_num)
        df.to_csv(feature_path, index=None)

    # 使用heartpy提取特征，并保存到csv文件中。
    def get_feature_from_heartpy(self):
        '''
        （1）使用heartpy提取特征，然后将提取的特征保存到csv文件中。
        （2）在训练树模型的时候，上述提取的特征接在renset模型提取的特征后面，训练树模型。
        '''
        logger.info("get heart feature use heartpy start...")
        patient_id = []
        df = pd.read_csv(args.train_label_csv)  # 注意要与get_feature_from_resnet中的文件路径一致
        for _, rows in df.iterrows():
            patient_id.append(rows['patient_id'])

        feature = []
        for _, patient in enumerate(tqdm(patient_id)):
            data_path = os.path.join(args.train_data_dir, patient + ".mat")
            ecg_data = sio.loadmat(data_path)['ecgdata']  # 'numpy.ndarray', (12, 5000)

            # 使用heartpy提取每一联的特征
            frequency = 500  # 心电图的采样率：500hz
            one_feature = [0] * 12 * 13
            for i in range(ecg_data.shape[0]):
                idx = i * 13
                try:
                    working_data, measures = hp.process(ecg_data[i], frequency)
                    # measures中有13个元素。
                    j = 0
                    for key in measures.keys():
                        index = idx + j
                        value = measures[key]
                        if np.isnan(float(value)):  # 去掉返回的字符
                            value = 0
                        one_feature[index] = value
                        j += 1
                except:
                    pass
                    # logger.info("Err: there is no heart feature: %s, %d" % (patient + ".mat", i))
            feature.append(one_feature)

        # write feature to cav
        df = pd.DataFrame(data=feature)
        df.to_csv(args.feature_heartpy_csv, index=None)

    # 将使用resnet模型和heartpy提取的特征合并到一个csv文件中
    def join_feature_resnet_heartpy(self):

        feature_heartpy = []
        df_heartpy = pd.read_csv(args.feature_heartpy_csv, sep='\t')
        for _, rows in df_heartpy.iterrows():
            rows = rows[0].split(",")
            feature_heartpy.append(rows)

        feature_all = []
        random_list = []
        df_resnet = pd.read_csv(args.feature_renet_csv, sep='\t')
        idx = 0
        for _, rows in df_resnet.iterrows():
            rows = rows[0].split(",")
            random = rows[-1]
            feature_resnet = rows[:-1]
            feature = feature_resnet + feature_heartpy[idx]
            random_list.append(random)
            feature_all.append(feature)

        # write feature to cav
        df = pd.DataFrame(data=feature_all)
        df['random'] = random_list
        df.to_csv(args.feature_train_csv, index=None)



if __name__ == "__main__":
    GetFeature().get_feature_from_heartpy()        # 使用heartpy抽取特征
    GetFeature().get_feature_from_resnet()          # 使用训练好的resnet网络抽取特征