# _*_ coding:utf-8 _*_

import os
import torch
import pandas as pd
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from utils.configures import args


class ECGDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, random):
        super(ECGDataset, self).__init__()
        self.phase = phase
        self.data_dir = data_dir
        df = pd.read_csv(label_csv)
        if random is not None:
            df = df[df['random'].isin(random)]
        self.labels = df
        self.leads = args.ecg_leads
        self.n_leads = args.num_leads
        self.classes = args.ecg_classes
        self.n_classes = len(self.classes)
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        patient_id = row['patient_id']
        data_path = os.path.join(self.data_dir, patient_id + ".mat")
        ecg_data = sio.loadmat(data_path)['ecgdata']                # ecg_data.shap:(12,5000), numpy.ndarray

        # ecg_data = self.transform(ecg_data, self.phase == 'train')          # ecg_data.shap:(12,5000)
        result = np.zeros((self.n_leads, 4800))                              # 10 s, 500 Hz
        result[:, :] = ecg_data[:, 100:-100]

        if self.phase == "test":
            labels = np.zeros(self.n_classes, dtype=np.int8)
        else:
            if self.label_dict.get(patient_id):
                labels = self.label_dict.get(patient_id)
            else:
                labels = row[self.classes].to_numpy(dtype=np.float32)
                self.label_dict[patient_id] = labels
        return torch.from_numpy(result).float(), torch.from_numpy(labels).float(), patient_id

    def __len__(self):
        return len(self.labels)

    def scaling_sig(self, sig, sigma=0.1):
        scal_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, sig.shape[1]))
        noise = np.matmul(np.ones((sig.shape[0], 1)), scal_factor)
        return sig * noise

    def shift_sig(self, sig, interval=20):
        for col in range(sig.shape[1]):
            offset = np.random.choice(range(-interval, interval))
            sig[:, col] += offset / 1000
        return sig

    def transform(self, sig, train=False):
        if train:
            if np.random.randn() > 0.65:
                sig = self.scaling_sig(sig)
            if np.random.randn() > 0.65:
                sig = self.shift_sig(sig)
        return sig
