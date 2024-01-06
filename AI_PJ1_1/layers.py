# coding=utf-8    # 代码文件编码方式

import numpy as np
import time

class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = 0-np.zeros([1, self.num_output])

    # 前向传播计算
    def forward(self, input):
        start_time = time.time()
        self.input = input
        # 全连接层的前向传播，计算输出结果
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    # 反向传播计算，top_diff是损失函数对输出的导数
    def backward(self, top_diff):
        # 计算参数梯度和本层损失
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True)
        # print(top_diff.shape)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    # 利用参数进行参数更新，lr是学习率
    def update_param(self, lr):
        self.weight -= lr * self.d_weight
        self.bias -= lr * self.d_bias

    # 加载参数，用于重新加载模型
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    # 用于保存参数，方便模型重新训练
    def save_param(self):
        return self.weight, self.bias

class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')

    # 计算前向传播的输出结果
    def forward(self, input):
        start_time = time.time()
        self.input = input
        # ReLU层的前向传播，计算输出结果
        output = np.maximum(0, input)
        return output

    # 计算反向传播输出的损失结果
    def backward(self, top_diff):
        # ReLU层的反向传播，计算本层损失
        bottom_diff = top_diff.copy()
        bottom_diff[self.input<0] = 0
        return bottom_diff

class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')

    # 计算前向传播的输出结果
    def forward(self, input):
        # softmax 损失层的前向传播，计算输出结果
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    # 计算损失
    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        # print(self.label_onehot)
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        # print("loss %.6f" % loss)
        return loss

    # 计算反向传播输出的损失结果
    def backward(self):
        # softmax 损失层的反向传播，计算本层损失
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff


