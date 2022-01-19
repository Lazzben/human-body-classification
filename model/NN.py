# -*- coding: utf-8 -*-
# @Time    : 2021/8/22 13:42
# @Author  : Huang Ben Hao
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model.BasicModule import BasicModule


class NN(BasicModule):
    def __init__(self, config: Config):
        """
        NN 模型
        :param config: 配置文件
        """
        super(NN, self).__init__()

        # self.dropout: float = config.dropout
        self.input_dim = config.input_dim
        self.output_dim = config.relation_type
        self.hidden_dim = config.hidden_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input_data):
        x = F.relu(self.fc1(input_data))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
