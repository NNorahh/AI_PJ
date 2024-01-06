# coding=utf-8
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from layers import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer

# 数据集的存储路径
DATA_DIR = "./train"

class classify_net(object):
    # 初始化函数，传入了多个参数用于初始化神经网络
    def __init__(self, batch_size=50, input_size=784, hidden=[128,128], out_classes=12, lr=0.001, max_epoch=100,
                 print_iter=10, mode="train"):
        self.batch_size = batch_size  # 每次训练迭代所使用的数据批次大小
        self.input_size = input_size  # 输入层神经元数量 28*28
        self.hidden_sizes = hidden
        self.out_classes = out_classes  # 输出层神经元数量（即分类数量）
        self.lr = lr  # 学习率，用于控制模型训练时每次参数的更新幅度
        self.max_epoch = max_epoch  # 最大训练轮数
        self.print_iter = print_iter  # 训练过程中每隔print_iter次输出一次日志信息
        self.accuracy_list = []  # 存储每次迭代的准确率
        self.loss_list = []  # 存储每次迭代损失
        self.mode = mode

    def load_data(self):
        if self.mode == "test":
            self.load_test_data()
            return
        print('加载数据集 ...')
        x = np.full((28, 28), 1/255)
        dataset = np.zeros((12 * 620, 28 * 28))
        ansSet = np.zeros((12 * 620, 1))
        ansSet = ansSet.astype(int)
        dataset = dataset.astype(int)
        index = 0
        for i in range(0, 12):
            for j in range(0, 620):
                filename = "train/%d/%d.bmp" % (i+1, j+1)
                img = cv2.imread(filename)
                imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result = imGray
                result = result.flatten()
                dataset[index] = result
                ansSet[index] = i
                index += 1
        if self.mode == "train":
            test_size = 0.1  # 验证集占总数据集的比例
            # 使用 train_test_split 函数划分数据集
            train_images, test_images, train_labels, test_labels = train_test_split(
                dataset, ansSet, test_size=test_size, random_state=42)
            self.train_data = np.append(train_images, train_labels, axis=1)
            self.test_data = np.append(test_images, test_labels, axis=1)


    # 将训练数据集打乱，以便更好地训练
    def shuffle_data(self):
        print('随机混洗数据...')
        np.random.shuffle(self.train_data)

    # 建立多层感知机神经网络，并定义各层的数量及其之间的连接关系
    def build_model(self):  # 建立网络结构
        print('构建模型...')

        self.fc_layers = []
        self.relu_layers = []

        # 构建全连接层和ReLU层
        for i in range(len(self.hidden_sizes)):
            if i == 0:
                input_size = self.input_size
            else:
                input_size = self.hidden_sizes[i - 1]
            hidden_size = self.hidden_sizes[i]
            fc_layer = FullyConnectedLayer(input_size, hidden_size)
            relu_layer = ReLULayer()
            self.fc_layers.append(fc_layer)
            self.relu_layers.append(relu_layer)

        # 输出层
        final_fc_layer = FullyConnectedLayer(self.hidden_sizes[-1], self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.final_fc = final_fc_layer
        # 将所有层组合成一个列表
        self.all_layers = self.fc_layers + self.relu_layers + [final_fc_layer, self.softmax]

        # 更新层列表
        self.update_layer_list = self.fc_layers + [final_fc_layer]

    # 初始化神经网络参数
    def init_model(self):
        # print('Initializing parameters of each layer in MLP...')
        print('初始化各层参数权值...')

        for layer in self.update_layer_list:
            layer.init_param()

    # 从文件加载神经网络参数（即权重和偏差）
    def load_model(self, param_dir):
        print('从文件加载参数权值：' + param_dir)
        params = np.load(param_dir, allow_pickle=True).item()

        for i, fc_layer in enumerate(self.update_layer_list):
            w_key = 'w' + str(i + 1)
            b_key = 'b' + str(i + 1)
            fc_layer.load_param(params[w_key], params[b_key])
        # print(params)


    def save_model(self, param_dir):
        print('保存权值和偏置到： ' + param_dir)
        params = {}

        for i, fc_layer in enumerate(self.update_layer_list):
            w_key = 'w' + str(i + 1)
            b_key = 'b' + str(i + 1)
            params[w_key], params[b_key] = fc_layer.save_param()
        np.save(param_dir, params)

    # 神经网络的前向传播，即将输入数据通过神经网络传递，得出预测结果
    def forward(self, input):  # 神经网络的前向传播
        h = input

        for fc_layer, relu_layer in zip(self.fc_layers, self.relu_layers):
            h = fc_layer.forward(h)
            h = relu_layer.forward(h)

        final_output = self.all_layers[-2].forward(h)  # 输出层
        prob = self.all_layers[-1].forward(final_output)  # 损失函数层
        return prob

    # 神经网络的反向传播，即求解梯度，调整神经网络参数
    def backward(self):  # 神经网络的反向传播
        dloss = self.softmax.backward()
        dh = self.final_fc.backward(dloss)
        for relu_layer, fc_layer in zip(reversed(self.relu_layers), reversed(self.fc_layers)):
            dh = relu_layer.backward(dh)
            dh = fc_layer.backward(dh)

    # 根据梯度和学习率更新神经网络参数
    def update(self, lr):  # 神经网络的参数更新
        for layer in self.update_layer_list:
            layer.update_param(lr)

    # 使用反向传播和梯度下降训练多层感知机神经网络，使其逐渐适应数据集
    def train(self):
        # 计算每个batch中的样本数
        max_batch = self.train_data.shape[0] // self.batch_size  # [0]表示数据集的第一维，也就是数据集中样本的数量
        print('开始训练...')
        # 对于每个epoch
        for idx_epoch in range(self.max_epoch):
            # 将训练数据集随机打乱
            self.shuffle_data()
            # 对于每个batch
            for idx_batch in range(max_batch):
                # 从训练数据中取出当前batch的图片和标签
                batch_images = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1]
                # 使用前向传播算法计算预测概率
                prob = self.forward(batch_images)
                # 使用Softmax交叉熵函数计算损失
                loss = self.softmax.get_loss(batch_labels)
                # 使用反向传播算法进行梯度下降更新模型参数
                self.backward()
                self.update(self.lr)
                # 如果当前batch的序号可以整除打印迭代步数的间隔，打印该batch的损失
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f print_iter: %d' % (idx_epoch, idx_batch, loss, self.print_iter))
                    #  计算当前batch的预测准确率并存储下来
                    pred_results = np.zeros([batch_labels.shape[0]])
                    for idx in range(batch_labels.shape[0] // self.batch_size):
                        # 从当前batch中取出样本进行预测
                        batch_images_temp = batch_images[idx * self.batch_size:(idx + 1) * self.batch_size, :]
                        prob = self.forward(batch_images_temp)
                        pred_labels = np.argmax(prob, axis=1)
                        pred_results[idx * self.batch_size:(idx + 1) * self.batch_size] = pred_labels
                    # 计算准确率
                    acc = np.mean(pred_results == batch_labels)
                    self.accuracy_list.append(acc)  # 存储当前batch的预测准确率
                    self.loss_list.append(loss)  # 存储当前batch的预测准确率

    def load_test_data(self):
        print('加载测试数据集 ...')
        x = np.full((28, 28), 1 / 255)
        test_dataset = np.zeros((12 * 240, 28 * 28))
        test_ansSet = np.zeros((12 * 240, 1))
        self.test_ans = np.zeros((12 * 240, 1))
        test_ansSet = test_ansSet.astype(int)
        test_dataset = test_dataset.astype(int)
        self.test_ans = self.test_ans.astype(int)
        index = 0
        for i in range(0, 12):
            for j in range(0, 240):
                filename = "test/%d/%d.bmp" % (i + 1, j + 1)
                img = cv2.imread(filename)
                imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result = imGray
                result = result.flatten()
                test_dataset[index] = result
                test_ansSet[index] = i
                index += 1

        self.test_data = np.append(test_dataset, test_ansSet, axis=1)

    # 在测试数据集上对神经网络进行推断，并计算分类准确率
    def evaluate(self):  # 推断函数
        pred_results = np.zeros([self.test_data.shape[0]])
        pred_results = pred_results.astype(int)
        for idx in range(self.test_data.shape[0] // self.batch_size):
            batch_images = self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx * self.batch_size:(idx + 1) * self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('测试集准确率: %f' % accuracy)


if __name__ == '__main__':
    e = 200
    mode = "train"
    model = classify_net(batch_size=40,hidden = [128,128], max_epoch=e, mode=mode)
    model.load_data()
    model.build_model()
    model.init_model()
    if mode == "train":
    # 训练：
        model.train()
        model.save_model('my_classify_model.npy')
        model.evaluate()
    elif mode == "test":
    #测试：
        model.load_model('my_classify_model.npy')
        model.evaluate()


