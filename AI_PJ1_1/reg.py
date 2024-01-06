import random
import numpy as np
import matplotlib.pyplot as plt

class reg_bp():
    def __init__(self, layer_dims, lr, accuracy, act_func):
        self.lr = lr
        self.accuracy = accuracy
        self.activation = act_func
        self.layer_dims = layer_dims
        self.layer_num = len(layer_dims)
        self.output_num = layer_dims[self.layer_num - 1]
        self.biases = []
        self.weights = []
        self.init_weights()
        self.init_biases()
        self.result = []

    def init_weights(self):
        for i in range(self.layer_num - 1):
            # initial weights for every layer
            layer_weights=[]
            for j in range(self.layer_dims[i + 1]):
                weights=[]
                for k in range(self.layer_dims[i]):
                    weights.append(random.random()*0.01)
                layer_weights.append(weights)
            self.weights.append(layer_weights)

    def init_biases(self):
        for i in range(self.layer_num - 1):
            # initial bias for every layer
            biases = []
            for j in range(self.layer_dims[i + 1]):
                biases.append(0-random.random())
            self.biases.append(biases)

    def forward(self, input_data):
        self.result = []
        self.result.append(input_data)
        hidden = np.dot(self.weights[0],input_data)
        for i in range(len(hidden)):
            hidden[i][0] += self.biases[0][i]
        if self.activation == 'sigmoid':
            hidden=sigmoid(hidden)
        elif self.activation == 'tanh':
            hidden=tanh(hidden)
        elif self.activation == 'relu':
            hidden=relu(hidden)
        self.result.append(hidden)
        for i in range(1, self.layer_num-1):
            hidden = np.dot(self.weights[i],hidden)
            for j in range(len(hidden)):
                hidden[j][0] += self.biases[i][j]
            if i != self.layer_num-2:#最后一层不用激活函数
                if self.activation == 'sigmoid':
                    hidden = sigmoid(hidden)
                elif self.activation == 'tanh':
                    hidden = tanh(hidden)
                elif self.activation == 'relu':
                    hidden = relu(hidden)
            self.result.append(hidden)


    def backward(self, output_data):
        residual_matrix = []
        for i in range(self.layer_num-2,-1,-1):#from the last-1 layer
            residual_matrix_element=[]
            if i == self.layer_num-2:
                for j in range(self.output_num):
                    result = [-(output_data[j][0] - self.result[i+1][j][0])]
                    residual_matrix_element.append(result)#the last layer has no activate func
                residual_matrix.append(residual_matrix_element)
            else:
                derivative_result = []
                for j in range(len(self.result[i+1])):
                    if self.activation == 'sigmoid':
                        element = [self.result[i + 1][j][0] * (1 - self.result[i + 1][j][0])]
                        derivative_result.append(element)
                    elif self.activation == 'tanh':
                        element = [1 - self.result[i+1][j][0]**2]
                        derivative_result.append(element)
                    elif self.activation == 'relu':
                        element = [1 if self.result[i + 1][j][0] > 0 else 0]
                        derivative_result.append(element)
                residual_matrix_element = np.dot(np.transpose(self.weights[i+1]), residual_matrix[self.layer_num-3-i])
                for j in range(len(derivative_result)):
                    residual_matrix_element[j][0] *= derivative_result[j][0]
                residual_matrix.append(residual_matrix_element)
        #adjust weights and biases
        for i in range(self.layer_num-2,-1,-1):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= (self.lr*residual_matrix[self.layer_num-2-i][j][0] * \
                                                    self.result[i][k][0])
        for i in range(self.layer_num-2,-1,-1):
            for j in range(len(self.biases[i])):
                self.biases[i][j] -= (self.lr * residual_matrix[self.layer_num-2-i][j][0])

    def predict(self, input_data):
        net.forward(input_data)
        return net.result[net.layer_num - 1][0]

    def train(self, input_data, output_data, iterations):
        for i in range(iterations):
            Error = 0
            for j in range(len(input_data) // self.layer_dims[0]):
                train_data = []
                expect_data = []
                for k in range(self.layer_dims[0]):
                    train_data.append(input_data[j * self.layer_dims[0] + k])
                for k in range(self.output_num):
                    expect_data.append(output_data[j * self.layer_dims[0] + k])
                self.forward(train_data)
                self.backward(expect_data)
                for k in range(self.output_num):
                    Error += abs(expect_data[k][0]-self.result[self.layer_num-1][k][0])
            if self.accuracy > Error / len(input_data):
                print(Error / len(input_data))
                break
            else:
                print( i,"误差:",Error / len(input_data))

#activate function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x,0)


net = reg_bp([1, 16, 32, 16, 1], 0.5, 0.005, "sigmoid")

x_data = np.random.uniform(-np.pi, np.pi, (1000, 1))
split_ratio = 0.8  # 80% train，20% test
split_index = int(x_data.shape[0] * split_ratio)
x_train = x_data[:split_index]
x_test = x_data[split_index:]
y_train = np.sin(x_train)
y_test = np.sin(x_test)

net.train(x_train, y_train, 2000)
#
predict = []
for j in range(len(x_test)):
    predict.append(net.predict([x_test[j]]))

err = 0
for k in range(len(y_test)):
    err += np.abs(np.mean((y_test[k] - predict[k])))
print("测试集均误差为：%.6f" % (err / len(x_test)))

plt.grid(True)
plt.scatter(x_test, y_test)
plt.scatter(x_test, predict)
plt.show()
