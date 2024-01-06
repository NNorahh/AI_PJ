import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from task3.BiLSTM_CRF import BiLSTM_CRF, get_data

EMBEDDING_DIM = 100
HIDDEN_DIM = 200
START_TAG = "<START>"
STOP_TAG = "<STOP>"

sorted_labels_chn = [
'O',
'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME'
, 'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT'
, 'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU'
, 'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE'
, 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG'
, 'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE'
, 'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO'
, 'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC'
]
# Load the dataset
sentences, tags = get_data('../NER/Chinese/train.txt')
# sentences, tags = get_data('NER/example_data/example_gold_result.txt')
states = sorted(set(sum(tags, [])))
observations = sorted(set(sum(sentences, [])))

# Create a dictionary mapping tags to indices
tag_to_ix = {tag: i for i, tag in enumerate(sorted_labels_chn)}
if '<START>' not in tag_to_ix:
    tag_to_ix['<START>'] = len(tag_to_ix)
if '<STOP>' not in tag_to_ix:
    tag_to_ix['<STOP>'] = len(tag_to_ix)
# Create a dictionary mapping words to indices
word_to_ix = {word: i for i, word in enumerate(observations)}
if '<UNK>' not in word_to_ix:
    word_to_ix['<UNK>'] = len(word_to_ix)

# Create the model
model = BiLSTM_CRF(len(observations)+1, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

# Define the loss function and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(1):
    for sentence, tag in zip(sentences, tags):
        model.zero_grad()
        sentence_in = torch.tensor([word_to_ix[word] for word in sentence], dtype=torch.long)
        targets = torch.tensor([tag_to_ix[t] for t in tag], dtype=torch.long)
        loss = model.neg_log_likelihood(sentence_in, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    print('Epoch: %d Loss: %.6f' % (epoch+1, loss.item()))

    # # Evaluate the model on the validation set
    # model.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for sentence, tags in zip(vadi_sentences, vadi_tags):
    #         sentence_in = torch.tensor([word_to_ix[word] for word in sentence], dtype=torch.long)
    #         targets = torch.tensor([tag_to_ix[tag] for tag in tags], dtype=torch.long)
    #         _, predicted = model(sentence_in)
    #         total += len(targets)
    #         correct += sum([1 for i, j in zip(predicted, targets) if i == j])
    #     print('Epoch: %d Accuracy: %.2f%%' % (epoch+1, 100 * correct / total))


f = open('Chinese3.pickle', 'wb')
pickle.dump(model, f)
f.close()
print("Model saved!")
