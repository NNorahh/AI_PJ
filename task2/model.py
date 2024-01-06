import numpy as np
from sklearn_crfsuite import CRF

def getData(path):
    word_lists = []
    tag_lists = []
    word_list = []
    tag_list = []
    fp = open(path, "r", encoding='utf-8')
    for line in fp:
        if line == "\n":
            word_lists.append(word_list)
            tag_lists.append(tag_list)
            tag_list = []
            word_list = []
            continue
        items = line.split()
        word, tag = items[0], items[1].rstrip()
        word_list.append(word)
        tag_list.append(tag)
    word_lists.append(word_list)
    tag_lists.append(tag_list)
    return word_lists, tag_lists

def word2Features(sent, i):
    word = sent[i]  # 当前词
    #希望捕捉到更长距离的依赖关系
    prev_word = '<s>' if i == 0 else sent[i - 1]  # 上一个词，如果当前词是第一个词，则用 '<s>' 表示起始标记（START_TAG）
    next_word = '</s>' if i == (len(sent) - 1) else sent[i + 1]  # 下一个词，如果当前词是最后一个词，则用 '</s>' 表示结束标记（STOP_TAG）
    prev_word2 = '<s>' if i <= 1 else sent[i - 2]  # 前两个词，如果当前词是前两个词之一，则用 '<s>' 表示起始标记（START_TAG）
    next_word2 = '</s>' if i >= (len(sent) - 2) else sent[i + 2]  # 后两个词，如果当前词是后两个词之一，则用 '</s>' 表示结束标记（STOP_TAG）

    features = {
        'w': word,  # 当前词的特征
        'w-1': prev_word,  # 上一个词的特征
        'w+1': next_word,  # 下一个词的特征
        'w-1:w': prev_word + word,  # 上一个词和当前词的组合特征
        'w:w+1': word + next_word,  # 当前词和下一个词的组合特征
        'w-1:w:w+1': prev_word + word + next_word,  # 上一个词、当前词、下一个词的组合特征
        'w-2:w': prev_word2 + word,  # 前两个词和当前词的组合特征
        'w:w+2': word + next_word2,  # 当前词和后两个词的组合特征
        'bias': 1  # 偏置项，用于引入一个额外的特征，帮助模型更好地适应训练数据
    }
    return features

def sent2Features(sent):
    return [word2Features(sent, i) for i in range(len(sent))]

class crf_ner(object):
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1,
                 max_iterations=100, all_possible_transitions=False):
        self.crf = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self, train_words, train_tags):
        features = [sent2Features(s) for s in train_words]
        self.crf.fit(features, train_tags)

    def val(self, val_words, out_path):
        f = open(out_path, "w+", encoding="utf-8")
        features = [sent2Features(s) for s in val_words]
        preds = self.crf.predict(features)
        for i, words in enumerate(val_words):
            for j in range(len(words)):
                f.write(words[j] + " " + preds[i][j] + "\n")
            if i!=len(val_words)-1:
                f.write("\n")
        f.close()