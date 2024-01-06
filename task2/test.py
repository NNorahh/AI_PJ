import model
import check

E_train_data = "../NER/English/train.txt"
E_test_data = "../NER/English/validation.txt"
E_output_path = "../NER/English/ans.txt"
C_train_data = "../NER/Chinese/train.txt"
C_test_data = "../NER/Chinese/validation.txt"
C_output_path = "../NER/Chinese/ans.txt"

C_tag_dict = {'O': 0, 'B-NAME': 1, 'M-NAME': 2, 'E-NAME': 3,
      'S-NAME': 4, 'B-CONT': 5, 'M-CONT': 6, 'E-CONT': 7, 'S-CONT': 8,
      'B-EDU': 9, 'M-EDU': 10, 'E-EDU': 11, 'S-EDU': 12, 'B-TITLE': 13,
      'M-TITLE': 14, 'E-TITLE': 15, 'S-TITLE': 16, 'B-ORG': 17,
      'M-ORG': 18, 'E-ORG': 19, 'S-ORG': 20, 'B-RACE': 21, 'M-RACE': 22,
      'E-RACE': 23, 'S-RACE': 24, 'B-PRO': 25, 'M-PRO': 26, 'E-PRO': 27,
      'S-PRO': 28, 'B-LOC': 29, 'M-LOC': 30, 'E-LOC': 31, 'S-LOC': 32}

E_tag_dict = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3,
      "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8}

# E_test_data = "./english_test.txt"
# C_test_data = "./chinese_test.txt"


# mode = "Chinese"
mode = "English"
if mode == "Chinese":
      train_path = C_train_data
      test_path = C_test_data
      tag_dict = C_tag_dict
      out_path = C_output_path
elif mode == "English":
      train_path = E_train_data
      test_path = E_test_data
      tag_dict = E_tag_dict
      out_path = E_output_path

# word2idx = model.word2idx([train_path, test_path])
# word_dict = model.GetDict([C_train_data, C_test_data])

train_words, train_tags = model.getData(train_path)
val_words, val_tags = model.getData(test_path)

crf = model.crf_ner()
print("training...")
crf.train(train_words, train_tags)
crf.val(val_words, out_path)
check.check(mode, gold_path=test_path, my_path=out_path)