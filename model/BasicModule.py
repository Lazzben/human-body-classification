# -*- coding: utf-8 -*-
# @Time    : 2021/8/22 13:38
# @Author  : Huang Ben Hao
import time

import torch.nn as nn
import torch

from utils import ensure_dir


class BasicModule(nn.Module):
    def _forward_unimplemented(self, *input) -> None:
        pass

    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name, epoch=0):
        prefix = 'checkpoints/'
        ensure_dir(prefix)

        name = prefix + '_' + name + "_" + f'epoch{epoch}_'

        name = time.strftime(name + '%m%d_%H_%M_%S.pth')

        torch.save(self.state_dict(), name)

        return name
