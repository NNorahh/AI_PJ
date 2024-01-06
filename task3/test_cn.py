import pickle
import torch
import BiLSTM_CRF
from task3.check import check
from task3.train_cn import word_to_ix, tag_to_ix

def write_result(result, read_path, write_path):
    with open(read_path, 'r', encoding='utf-8') as fr, open(write_path, 'w', encoding='utf-8') as fw:
        idx = 0
        line = fr.readline()
        while line:
            if line == '\n':
                fw.write(line)
            else:
                word, _ = line.split()
                predict_line = word + ' ' + result[idx]
                fw.write(predict_line)
                fw.write('\n')
                idx += 1
            line = fr.readline()

f = open('Chinese3.pickle', 'rb')
model = pickle.load(f)
f.close()

vadi_sentences, vadi_tags = BiLSTM_CRF.get_data('../NER/Chinese/validation.txt')

with torch.no_grad():
    predicted_tags = []
    for sentence, tags in zip(vadi_sentences, vadi_tags):
        sentence_in = torch.tensor([word_to_ix.get(word, word_to_ix['<UNK>']) for word in sentence], dtype=torch.long)
        targets = torch.tensor([tag_to_ix[tag] for tag in tags], dtype=torch.long)
        _, pred_tags = model(sentence_in)
        predicted_tags += [list(tag_to_ix.keys())[list(tag_to_ix.values()).index(tag)] for tag in pred_tags]

write_result(predicted_tags, '../NER/Chinese/validation.txt', '../NER/Chinese/ans.txt')
check(language = "Chinese", gold_path="../NER/Chinese/validation.txt", my_path="../NER/Chinese/ans.txt")