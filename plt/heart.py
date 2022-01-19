#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/9 上午11:31
# @Author : HuangBenHao
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

df_1 = pd.read_excel('./数据采集表-1.11.xlsx', sheet_name='正手快攻').dropna(axis=1)
del df_1['Unnamed: 0']
df_2 = pd.read_excel('./数据采集表-1.11.xlsx', sheet_name='反手快拨').dropna(axis=1)
del df_2['Unnamed: 0']
df_3 = pd.read_excel('./数据采集表-1.11.xlsx', sheet_name='反手搓球').dropna(axis=1)
del df_3['Unnamed: 0']
df_4 = pd.read_excel('./数据采集表-1.11.xlsx', sheet_name='前冲弧圈').dropna(axis=1)
del df_4['Unnamed: 0']
df_5 = pd.read_excel('./数据采集表-1.11.xlsx', sheet_name='加转弧圈').dropna(axis=1)
del df_5['Unnamed: 0']
df_6 = pd.read_excel('./数据采集表-1.11.xlsx', sheet_name='对抗比赛').dropna(axis=1)
del df_6['Unnamed: 0']

ax = plt.subplot(321)
seaborn.boxplot(data=df_1)
plt.title("正手快攻")


plt.subplot(322)
seaborn.boxplot(data=df_2)
plt.title("反手快拨")

plt.subplot(323)
seaborn.boxplot(data=df_3)
plt.title("反手搓球")

plt.subplot(324)
seaborn.boxplot(data=df_4)
plt.title("前冲弧圈")

plt.subplot(325)
plt.title("加转弧圈")
seaborn.boxplot(data=df_5)

plt.subplot(326)
plt.title("对抗比赛")
seaborn.boxplot(data=df_6)


plt.show()
