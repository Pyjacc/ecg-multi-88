# _*_ coding:utf-8 _*_

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from models.resnet import resnet34
from utils.dataset import ECGDataset
from utils.tools import tools
from utils.tools import logger
from utils.configures import args

tools.seed_torch(args.seed)

def get_thresholds(val_loader, net, device):
    logger.info('finding 12 leads optimal thresholds')

    output_list, label_list = [], []
    for _, (data, label) in enumerate(tqdm(val_loader)):
        data, labels = data.to(device), label.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())

    y_trues = np.vstack(label_list)
    y_hats = np.vstack(output_list)
    thresholds = []

    for i in range(y_trues.shape[1]):
        y_true = y_trues[:, i]
        y_hat = y_hats[:, i]
        threshold = tools.find_optimal_threshold(y_true, y_hat)
        thresholds.append(threshold)

    return thresholds


def plot_confusion_matrix(y_trues, y_preds, normalize=True, cmap=plt.cm.Blues):
    for i, label in enumerate(args.ecg_classes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=[0, 1], yticklabels=[0, 1],
               title=label,
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), ha="center")

        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        np.set_printoptions(precision=3)
        fig.tight_layout()
        plt.savefig(args.png_dir + f'/{label}.png')
        plt.close(fig)


def predict_use_thresholds(data_loader, net, device, thresholds):
    output_list, label_list = [], []

    for _, (data, label) in enumerate(tqdm(data_loader)):
        data, labels = data.to(device), label.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())

    y_trues = np.vstack(label_list)     # y_trues.shape:(2800, 12)
    y_hats = np.vstack(output_list)     # y_hats.shape:(2800, 12)

    y_preds = []
    scores = []

    for i in range(len(thresholds)):
        y_true = y_trues[:, i]
        y_hat = y_hats[:, i]
        y_pred = (y_hat >= thresholds[i]).astype(int)
        scores.append(tools.metrics_scores(y_true, y_pred, y_hat))
        y_preds.append(y_pred)

    y_preds = np.array(y_preds).transpose()
    scores = np.array(scores)
    print('Precisions:', scores[:, 0])
    print('Recalls:', scores[:, 1])
    print('F1s:', scores[:, 2])
    print('AUCs:', scores[:, 3])
    print('Accs:', scores[:, 4])
    print(np.mean(scores, axis=0))

    if args.platform == 'local':
        plot_confusion_matrix(y_trues, y_preds)

def predict(data_loader, net, device, thresholds):
    output_list = []

    for _, (data, label) in enumerate(tqdm(data_loader)):
        data = data.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())

    y_hats = np.vstack(output_list)     # y_hats.shape:(2800, 12)
    y_preds = []

    for i in range(len(thresholds)):
        y_hat = y_hats[:, i]
        y_pred = (y_hat >= thresholds[i]).astype(int)
        y_preds.append(y_pred)

    y_preds = np.array(y_preds).transpose()

    # write predict result to answer.csv




if __name__ == "__main__":
    # 以下两行路径设置用于在aiwini平台上找到预测时12导联的阈值。
    # args.test_data_dir = args.train_data_dir
    # args.test_label_csv = args.train_label_csv

    device = tools.get_device(args)
    data_dir = os.path.normpath(args.test_data_dir)

    test_random_num, val_random_num = tools.split_train_dev_data(seed=args.seed)
    val_dataset = ECGDataset('val', data_dir, args.test_label_csv, val_random_num)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataset = ECGDataset('test', data_dir, args.test_label_csv, test_random_num)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = torch.load(args.model_path, map_location=device)
    num_class = len(args.ecg_classes)
    num_leads = args.num_leads
    network = resnet34(input_channels=num_leads, num_classes=num_class).to(device)
    network.load_state_dict(model)
    network.eval()

    thresholds = get_thresholds(val_loader, network, device)
    print('12 leads thresholds: ', thresholds)

    logger.info('results on test data:')
    predict_use_thresholds(test_loader, network, device, thresholds)
    # predict(test_loader, network, device, thresholds)
