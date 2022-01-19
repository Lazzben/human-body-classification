#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/1/5 下午3:36
# @Author : HuangBenHao
from flask import Flask, request, jsonify
from predict2 import predict
import pandas as pd

app = Flask(__name__)


@app.route('/cal/stature', methods=['post'])
def hello_world():
    data = eval(request.data.decode('utf-8'))
    stature = predict([[float(i) for i in data['data']]])
    return str(stature)


def get_heart_rate_date():
    return pd.read_excel('./data/hr/1.xlsx')


@app.route('/cal/heartrate', methods=['get'])
def cal_heart_rate():
    df = get_heart_rate_date()
    data_dict = df.describe().to_dict()
    result = []
    for key in data_dict:
        data_dict[key]['name'] = key
        data_dict[key]['line25'] = data_dict[key]['25%']
        data_dict[key]['line50'] = data_dict[key]['50%']
        data_dict[key]['line75'] = data_dict[key]['75%']
        result.append(data_dict[key])
    return jsonify({'data': result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
