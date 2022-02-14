#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/14 下午4:40
# @Author : HuangBenHao
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

confusion_matrix = []
with open('./confusion_matrix.txt', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.replace(']', '').replace('[', '').strip().split(' ')
        line = [int(i) for i in line if i != '']
        confusion_matrix.append(line)

temp = {
    '低体重高脂肪': 0,
    '低体重': 1,
    '低脂肪': 2,
    '脂肪过多': 3,
    '标准': 4,
    '低脂肪肌肉型': 5,
    '肥胖': 6,
    '肥胖临界': 7,
    '超重': 8,
    '肌肉型超重': 9,
    '临界线': 10
}

label = [i for i in temp.keys()]


def draw_confusion_matrix(confusion_matrix, label):
    plt.figure(figsize=(10, 8), dpi=100)
    ax = sns.heatmap(pd.DataFrame(confusion_matrix),
                     square=True,
                     annot=True,
                     fmt='.20g',
                     linewidths=.5,
                     cmap="Blues",
                     xticklabels=label,
                     yticklabels=label)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.show()


draw_confusion_matrix(confusion_matrix, label)
