# -*- coding: utf-8 -*-
# @Time    : 2021/8/22 13:29
# @Author  : Huang Ben Hao
class Config(object):
    test_data_path = 'data/origin/test_1109.txt'
    train_data_path = 'data/origin/train_1109.txt'
    data_out = 'data/out'
    model_name = 'NN'

    seed = 123
    use_gpu = True
    gpu_id = 0

    batch_size = 512
    learning_rate = 0.01
    epoch = 100
    decay_rate = 0.3
    decay_patience = 5
    # decay_rate = 0
    # decay_patience = 0

    input_dim = 12
    relation_type = 11
    hidden_dim = 128


config = Config()
