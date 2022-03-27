# _*_ coding:utf-8 _*_

'''训练第一个模型，用于判断心电图是others还是非others'''

import os
import torch
from torch.utils.data import DataLoader
from models.resnet.resnet34_rdrop import resnet342d7_15
from utils.configures import args
from steps.other.dataset_other import ECGDataset        # todo: 需要根据训练的模型不同而进行调整
from utils.tools import tools
from utils.tools import logger
from utils.processor import train_ecg, evaluate_ecg
from utils.preprocess import get_train_dev_test_label_others_local, split_train_dev_test_data_others, get_train_label_aiwin

tools.seed_torch(args.seed)

def train_model():
    data_dir = os.path.normpath(args.train_data_dir)
    device = tools.get_device(args)

    train_random_num = tools.get_random_num(seed=args.seed)
    dev_random_num = tools.get_random_num(seed=args.seed)
    train_dataset = ECGDataset('train', data_dir, args.train_label_others_csv, train_random_num)     # todo:传入对应的train_label_csv
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dev_dataset = ECGDataset('val', data_dir, args.dev_label_others_csv, dev_random_num)             # todo:传入对应的dev_label_csv
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_class = len(args.ecg_classes_others)        # todo: 需要根据训练的模型不同而进行调整
    num_leads = args.num_leads
    network = resnet342d7_15(input_channels=num_leads, num_classes=num_class).to(device)  # todo: 选择不同的模型
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    criticism = torch.nn.BCEWithLogitsLoss()

    # train model
    args.best_f1 = 0
    args.best_epoch = 0
    model_type = 'others_model_2d'     # todo
    for epoch in range(args.epochs):
        logger.info("===========trainin===========")
        train_ecg(train_loader, network, criticism, device, epoch, scheduler, optimizer)
        logger.info("===========evaluation===========")
        evaluate_ecg(dev_loader, network, criticism, device, epoch, model_type)

    logger.info("overall best epoch: %d, f1: %.4f" % (args.best_epoch, args.best_f1))


if __name__ == "__main__":
    logger.info("get train and test data labels")

    if args.platform == 'local':
        split_train_dev_test_data_others()             # split test data and train data
        get_train_dev_test_label_others_local()         # get train and test labels from trainreference.csv
    else:
        get_train_label_aiwin()

    logger.info("start train model...")
    train_model()
