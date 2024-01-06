import model
import check

E_train_data = "../NER/English/train.txt"
E_test_data = "../NER/English/validation.txt"
E_output_path = "../NER/English/ans.txt"
C_train_data = "../NER/Chinese/train.txt"
C_test_data = "../NER/Chinese/validation.txt"
C_output_path = "../NER/Chinese/ans.txt"

# E_test_data = "./english_test.txt"
# C_test_data = "./chinese_test.txt"

model = model.HMM()

# mode = "Chinese"
mode = "English"
if mode == "Chinese":
      train_data = C_train_data
      test_data = C_test_data
      output_path = C_output_path
elif mode == "English":
      train_data = E_train_data
      test_data = E_test_data
      output_path = E_output_path

print("training......")
model.train(train_data)
model.val(test_data, output_path)
check.check(mode, test_data, output_path)