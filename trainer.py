# -*- coding: utf-8 -*-
# @Time    : 2021/8/22 13:59
# @Author  : Huang Ben Hao
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import warnings


def train(epoch: int, data_loader: DataLoader, device, model: nn.Module, optimizer, loss_fun):
    """
    训练
    :param epoch:
    :param data_loader:
    :param device: cpu or gpu
    :param model: 模型
    :param optimizer: 优化器
    :param loss_fun: 损失函数
    :return:
    """
    global idx
    print('*' * 20, 'epoch', epoch, '*' * 20)
    total_loss = []
    count = 0
    acc_all = 0
    for batch_idx, batch in enumerate(data_loader):
        # train
        model.train()
        x, y = [data.to(device) for data in batch]
        y_hat = model(x)

        optimizer.zero_grad()
        loss = loss_fun(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        # 日志
        correct_pred = torch.sum(torch.argmax(y_hat, dim=1) == y).item()
        acc_all += correct_pred
        count += len(y)

        if (batch_idx % 10 == 0) or (batch_idx == len(data_loader) - 1):
            print('Train epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'
                  .format(epoch, count, len(data_loader.dataset), 100. * count / len(data_loader.dataset), loss.item()))

    print('Train epoch:{}\t Average_Loss:{:.6f}\tACC:{:.2f}%'.format(epoch, sum(total_loss) / len(total_loss),
                                                                     acc_all / count * 100.))

    return sum(total_loss) / len(total_loss), acc_all / count


def validate(data_loader: DataLoader, device, model: nn.Module):
    """
    验证
    :param data_loader:
    :param device:
    :param model:
    :return: [macro_f1_score:float, micro_f1_score:float]
    """
    model.eval()
    total_f1 = []
    with torch.no_grad():
        total_y_pred = np.empty(0)
        total_y_true = np.empty(0)
        for batch_idx, batch in enumerate(data_loader):
            x, y = [data.to(device) for data in batch]
            y_pred = model(x)
            y_pred = y_pred.argmax(dim=-1)

            # 使用 precision_recall_fscore_support 必须在cpu上进行，且需要将数据转换成 numpy 格式。
            try:
                y_true, y_pred = y.numpy(), y_pred.numpy()
            except:
                y_true, y_pred = y.cpu().numpy(), y_pred.cpu().numpy()

            total_y_pred = np.append(total_y_pred, y_pred)
            total_y_true = np.append(total_y_true, y_true)

        for average in ['macro', 'micro']:
            warnings.filterwarnings("ignore")
            p, r, f1, s = precision_recall_fscore_support(total_y_true, total_y_pred, average=average)
            print(confusion_matrix(total_y_true, total_y_pred))
            print(f'{average} metrics:[precision:{p:.4f}, recall:{r:.4f}, f1:{f1:.4f}]')
            total_f1.append(f1)

        return total_f1
