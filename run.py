# -*- coding: utf-8 -*-
# @Time    : 2021/8/22 13:34
# @Author  : Huang Ben Hao
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from visdom import Visdom

from config import config
from dataset import CustomDataset, collate_fn
from model.NN import NN
from trainer import validate, train
from utils import make_seed

__Modules__ = {
    'NN': NN
}

# 命令行工具
parser = argparse.ArgumentParser(description='人体体质分类')
parser.add_argument('--model_name', type=str, default='NN', help="model name")
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model_name if args.model_name else config.model_name
    make_seed(config.seed)

    if config.use_gpu and torch.cuda.is_available():
        device = f"cuda:{config.gpu_id}"
    else:
        device = "cpu"

    device = torch.device(device)

    model = __Modules__[config.model_name](config)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=config.decay_rate,
                                                     patience=config.decay_patience)
    loss_fun = nn.CrossEntropyLoss()

    best_macro_f1, best_macro_epoch = 0, 1
    best_micro_f1, best_micro_epoch = 0, 1
    best_macro_model, best_micro_model = '', ''

    train_dataset = CustomDataset(config.train_data_path)
    test_dataset = CustomDataset(config.test_data_path)

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    print('*' * 20, '开始训练', '*' * 20)

    wind = Visdom()

    wind.line([[1., 0.]], [0], win='train', opts=dict(title='loss&acc', legend=['loss', 'acc']))

    for epoch in range(1, config.epoch + 1):
        loss, Acc = train(epoch, train_data_loader, device, model, optimizer, loss_fun)
        wind.line([[loss, Acc]], [epoch], win='train', update='append')
        macro_f1, micro_f1 = validate(test_data_loader, device, model)
        model_name = model.save(epoch=epoch, name=config.model_name)
        # 能够更好的学到一个局部最优解，训练的步子会越来越小
        scheduler.step(macro_f1)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_macro_epoch = epoch
            best_macro_model = model_name

        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            best_micro_epoch = epoch
            best_micro_model = model_name

    print('*' * 20, '结束训练', '*' * 20)

    print(f'best macro f1:{best_macro_f1:.4f}', f'best macro epoch:{best_macro_epoch}', f'save in {best_macro_model}')
    print(f'best micro f1:{best_micro_f1:.4f}', f'best micro epoch:{best_micro_epoch}', f'save in {best_micro_model}')
