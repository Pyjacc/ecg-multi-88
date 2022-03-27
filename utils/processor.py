# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 12:29
# @Author  : Qinglong
# @File    : processor.py.py
# @Description: In User Settings Edit

from tqdm import tqdm
import torch
import numpy as np
from utils.tools import tools
from utils.tools import logger
from utils.configures import args


def train_ecg(dataloader, network, criterion, device, epoch, scheduler, optimizer):
    network.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels, patientid) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = network(data)
        loss_kl = output[1]
        output = output[0]
        loss = criterion(output, labels)
        loss = loss + 10 * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())

    # scheduler.step()          # 设置学习率随着epoch改变
    logger.info('training, epoch: %d, loss: %.4f' % (epoch, running_loss))

def evaluate_ecg(dataloader, network, criterion, device, epoch, model_type='usual'):
    network.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels, patientid) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = network(data)
        loss_kl = output[1]
        output = output[0]
        loss = criterion(output, labels)
        loss = loss + 10 * loss_kl

        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())

    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = tools.metrics_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    logger.info("validating, loss: %.4f, f1: %.4f" % (running_loss, avg_f1))

    if args.phase == 'train' and avg_f1 > args.best_f1:
        logger.info("save the best model, model type: %s, f1: %.4f" % (model_type, avg_f1))

        if model_type == 'normal_model_1d':
            torch.save(network.state_dict(), args.model_path_normal_1d)        # 正常和非正常模型的保存路径
        elif model_type == 'normal_model_2d':
            torch.save(network.state_dict(), args.model_path_normal_2d)
        elif model_type == 'others_model_1d':
            torch.save(network.state_dict(), args.model_path_others_1d)     # 用于判断是othrs还是normal
        elif model_type == 'others_model_2d':
            torch.save(network.state_dict(), args.model_path_others_2d)
        elif model_type == 'resnet_1d':
            torch.save(network.state_dict(), args.model_path_1d)            # 1维卷积核模型
        elif model_type == 'resnet_1d_3_5':
            torch.save(network.state_dict(), args.model_path_1d_3_5)        # 卷积核为3*3和5*5的1维卷积模型
        elif model_type == 'resnet_1d_7_15':
            torch.save(network.state_dict(), args.model_path_1d_7_15)
        elif model_type == 'resnet_2d':
            torch.save(network.state_dict(), args.model_path_2d)            # 2维卷积核模型
        elif model_type == 'resnet_2d_3_5':
            torch.save(network.state_dict(), args.model_path_2d_3_5)
        elif model_type == 'resnet_2d_7_15':
            torch.save(network.state_dict(), args.model_path_2d_7_15)
        else:
            torch.save(network.state_dict(), args.model_path)

        args.best_f1 = avg_f1
        args.best_epoch = epoch
