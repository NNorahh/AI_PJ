import numpy as np


class HMM():

    def __init__(self):
        self.train_path = ""
        #字 转换为 token
        self.word2idx = {}
        #标签 转换为 token
        self.tag2idx = {}
        # token 转换为 标签
        self.idx2tag = {}
        self.id2word = {}
        self.word_num = 0
        self.tag_num = 0
        #初始概率矩阵、发射概率矩阵、状态转移概率矩阵
        self.init = None
        self.emit = None
        self.trans = None

    def genIdx(self):
        for line in open(self.train_path, "r", encoding='utf-8'):
            if line == "\n":
                continue
            items = line.split()
            word, tag = items[0], items[1].rstrip()
            #新的单词/汉字--->转化为idx
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.id2word[len(self.id2word)] = word
            #新的tag--->转化为idx
            if tag not in self.tag2idx:
                self.tag2idx[tag] = len(self.tag2idx)
                self.idx2tag[len(self.idx2tag)] = tag
        #防止val、test集有没出现过词
        self.word2idx['<unk>'] = len(self.word2idx)
        self.id2word[len(self.id2word)] = '<unk>'
        self.word_num = len(self.word2idx)
        self.tag_num = len(self.tag2idx)

    def genPro(self):
        self.init = np.zeros(self.tag_num)
        self.trans = np.zeros((self.tag_num, self.tag_num))
        self.emit = np.zeros((self.tag_num, self.word_num))

        #根据前一个标签，更新初始概率、发射概率和转移概率的统计信息
        #pre_tag = “”代表当前为初始状态
        pre_tag = ""
        for line in open(self.train_path, 'r', encoding='utf-8'):
            if line == "\n":
                pre_tag = ""
                continue
            items = line.split()
            word, token= items[0], items[1].rstrip()
            wid, tid = self.word2idx[word], self.tag2idx[token]
            #统计
            if pre_tag == "":
                self.init[tid] += 1#隐状态出现次数+1
                self.emit[tid][wid] += 1#t->w发射次数+1
            else:
                self.emit[tid][wid] += 1
                ptid = self.tag2idx[pre_tag]
                self.trans[ptid][tid] += 1#隐状态转移次数+1

            pre_tag = token

        #计算三个概率矩阵
        self.init = self.init / sum(self.init)
        for i in range(self.tag_num):
            self.trans[i] /= sum(self.trans[i])
            self.emit[i] /= sum(self.emit[i])


    def train(self, train_path):
        self.train_path = train_path
        self.genIdx()
        self.genPro()

    #采用对数概率，这样源空间中的很小概率，就被映射到对数空间的大的负数，同时相乘操作也变成简单的相加操作
    #如果某元素没有出现过，矩阵中的元素为0，这在后续的计算中是不允许的, 所以加上一个很小的数字
    def log(self, v):
        if v == 0:
            return np.log(v + 1e-10)
        return np.log(v)

    def viterbi(self, words_seq):
        seq_len = len(words_seq)
        words= [0] * seq_len
        for i in range(seq_len):
            words[i] = self.word2idx.get(words_seq[i], self.word2idx['<unk>'])
        #     words_seq[i] = self.word2idx.get(word, self.word2idx['<unk>'])

        viterbi = np.zeros((seq_len, self.tag_num))#时刻->状态
        backpointer = np.zeros((seq_len, self.tag_num), dtype=int)

        #t=1
        for j in range(self.tag_num):
            viterbi[0][j] = self.log(self.init[j]) + self.log(self.emit[j][words[0]])

        for i in range(1, seq_len):#对于t=2开始的每个时刻
            for j in range(self.tag_num):#对于每个隐藏状态
                #初始化路径长度为无限长
                viterbi[i][j] = -999999
                for k in range(self.tag_num):
                    score = viterbi[i - 1][k] + self.log(self.trans[k][j]) + self.log(self.emit[j][words[i]])
                    #找这个时刻的最优路径
                    if score > viterbi[i][j]:
                        viterbi[i][j] = score
                        backpointer[i][j] = k
                        
        best_tag = [0] * seq_len
        best_tag[-1] = np.argmax(viterbi[-1])
        tag_list = []
        #倒推路径
        for i in range(seq_len - 2, -1, -1):
            best_tag[i] = backpointer[i + 1][best_tag[i + 1]]
        for i in range(len(best_tag)):
            tag_list.append(self.idx2tag[best_tag[i]])

        return tag_list

    def val(self, test_path, output_path):
        ifp = open(test_path, "r", encoding='utf-8')
        ofp = open(output_path, 'w+', encoding='utf-8')
        word_seq = []
        tag_list = []
        for line in ifp:
            if line == "\n":
                pred = self.viterbi(word_seq)
                for i in range(len(tag_list)):
                    ofp.write(word_seq[i] + " " + pred[i] + '\n')
                ofp.write('\n')
                tag_list.clear()
                word_seq.clear()
                continue

            items = line.split()
            word, sym = items[0], items[1].rstrip()
            word_seq.append(word)
            tag_list.append(sym)

        if word_seq.__len__() != 0:
            pred = self.viterbi(word_seq)
            for i in range(len(tag_list)):
                ofp.write(word_seq[i] + " " + pred[i] + '\n')

        ifp.close()
        ofp.close()

