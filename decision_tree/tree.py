#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/14 下午3:07
# @Author : HuangBenHao

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_excel('../knn/sampling_data_1109.xlsx')

y = data['label_idx']
del data['label_idx']

y = y.values
x = data.values

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.7)

model = tree.DecisionTreeClassifier(
    criterion='gini',
    splitter='random',
    max_depth=11,
    class_weight='balanced',
)

clf = model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
