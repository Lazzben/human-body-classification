# -*- coding: utf-8 -*-
# @Time    : 2021/8/22 13:27
# @Author  : Huang Ben Hao
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.file = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line != '':
                    line = [float(i) for i in line.split(',')]
                    self.file.append(line)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample

    def __len__(self):
        return len(self.file)


def collate_fn(batch: list):
    """
    :param batch: 一个 batch 数据
    :return: 转化为tensor格式
    """
    data_list = []
    label_list = []

    for data in batch:
        float_data = [float(i) for i in data]
        data_list.append(float_data[:-1])
        label_list.append(int(float_data[-1]))

    return torch.tensor(data_list), torch.tensor(label_list)
