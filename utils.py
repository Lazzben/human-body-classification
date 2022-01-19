# -*- coding: utf-8 -*-
# @Time    : 2021/8/22 13:40
# @Author  : Huang Ben Hao
import torch
import numpy as np
import random

import os


def ensure_dir(directory):
    """
    判断目录是否存在，不存咋就创建
    :param directory:
    :return:
    """
    if not os.path.exists(directory):
        os.mkdir(directory)


def make_seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)
