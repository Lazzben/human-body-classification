#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/23 下午3:51
# @Author : HuangBenHao
from config import config
from model.NN import NN
import numpy as np
import torch
import joblib

if __name__ == '__main__':
    model = NN(config)
    model.load_state_dict(
        torch.load(r'/Users/lazyben/Downloads/human_body_classification/checkpoints/_NN_epoch88_1109_16_19_13.pth'))

    scaler = joblib.load(r'/Users/lazyben/Desktop/0822人体打标/1109/min_max_scaler.pkl')

    data = [[0., 18., 175., 53.9, 46.1, 49.3, 4.6, 8.5,
             17.6, 11.2, 65.1, 1733.3]]

    data = scaler.transform(data).astype(np.float32)
    print(torch.argmax(model(torch.tensor(data))).item())
