import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import nltk
import codecs
import numpy as np

import pickle

from bi_1_word_embed import ClassifyMovie
from snli_load_bigram import prepSNLI

EMBEDDING_DIM = 50

train, test, train_bigram, test_bigram, vocab = prepSNLI()

# random.shuffle(train)

print(len(vocab))
word_to_ix = {word: i for i, word in enumerate(vocab)}
print(len(word_to_ix))

loss_fucntion = nn.NLLLoss()


loss_fucntion = nn.NLLLoss()

def train(train_data, bigram_data, model):
    
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # for epoch in range(4):
    model.train()
    total_loss = torch.cuda.FloatTensor([0])
    skip_count = 0
    for sentence_label, bigrams_label in zip(train_data, bigram_data):
        # first_data = bigram_data[0]
        # sentence = first_data[0]
        # label = first_data[1]
        sentence = sentence_label[0]
        label = sentence_label[1]
        bigrams = bigrams_label[0]
        # print(sentence)
        # print(bigrams)
        # print(label)
        # print()
        # print()
        
        if (len(bigrams) == 0):
            skip_count += 1
            continue

        # print(idx_i)
        # idx_i += 1

        model.hidden = model.init_hidden()
        model.zero_grad()

        bigram_list = []
        for bigram in bigrams:
            bigram_idxs = [word_to_ix[w] for w in bigram]
            bigram_list.append(bigram_idxs)

        bigram_list_var = autograd.Variable(torch.cuda.LongTensor(bigram_list))
        # offsets = Variable(torch.cuda.LongTensor([0]))

        # print("bigram_list_var ", bigram_list_var)

        sentence_idxs = [word_to_ix[w] for w in sentence]
        sentence_var = autograd.Variable(torch.cuda.LongTensor(sentence_idxs))

        offsets = Variable(torch.cuda.LongTensor([0]))

        log_probs = model(sentence_var, bigram_list_var, offsets)
        # print(log_probs)

        loss = loss_fucntion(log_probs, autograd.Variable(torch.cuda.LongTensor([label])))

        loss.backward()
        optimizer.step()

        total_loss += (loss.data/(len(train_data)-skip_count))
    losses.append(total_loss.cpu().numpy()[0])
    return total_loss



def val(val_data, bigram_data, model):
    print("***Validation***")

    correct_count = 0
    val_loss = torch.cuda.FloatTensor([0])

    model.eval()
    skip_count = 0

    for sentence_label, bigrams_label in zip(val_data, bigram_data):

        sentence = sentence_label[0]
        label = sentence_label[1]
        bigrams = bigrams_label[0]
        
        if (len(bigrams) == 0):
            skip_count += 1
            continue


        model.hidden = model.init_hidden()

        bigram_list = []
        for bigram in bigrams:
            bigram_idxs = [word_to_ix[w] for w in bigram]
            bigram_list.append(bigram_idxs)

        bigram_list_var = autograd.Variable(torch.cuda.LongTensor(bigram_list))

        sentence_idxs = [word_to_ix[w] for w in sentence]
        sentence_var = autograd.Variable(torch.cuda.LongTensor(sentence_idxs))

        offsets = Variable(torch.cuda.LongTensor([0]))

        log_probs = model(sentence_var, bigram_list_var, offsets)
        # print(log_probs)

        loss = loss_fucntion(log_probs, autograd.Variable(torch.cuda.LongTensor([label])))
        _, idx = torch.max(log_probs, 1)
        # print(idx)
        # print(label)
        val_loss += (loss.data/(len(val_data)-skip_count))

        if idx.data.cpu().numpy() == label:
            correct_count += 1
        # print(correct_count)

    print("skip_count ", skip_count)
    print("Accuracy  = ", correct_count*100./(len(val_data)-skip_count))
    print("Accuracy(nllloss)  = ",val_loss[0])


cv_list = np.arange(0.2, 0.8, 0.2)
for i in range(3):

    print("\ninstance : ", str(i+2))
    val_data_cv = data[int(len(data)*cv_list[i]):int(len(data)*cv_list[i+1])]
    train_data_cv = data[:int(len(data)*cv_list[i])] + data[int(len(data)*cv_list[i+1]):]

    bigram_val_data_cv = bigram_data[int(len(bigram_data)*cv_list[i]):int(len(bigram_data)*cv_list[i+1])]
    bigram_train_data_cv = bigram_data[:int(len(bigram_data)*cv_list[i])] + bigram_data[int(len(bigram_data)*cv_list[i+1]):]

    model = ClassifyMovie(len(word_to_ix), EMBEDDING_DIM, 2)
    # model.load_embed(word_to_ix)
    model.cuda()

    for epoch in range(10):

        total_loss = train(train_data_cv, bigram_train_data_cv,  model)
        print("\nepoch " + str(epoch) + " " + str(total_loss[0]))

        # model.load_state_dict(torch.load('model/bigram_1'))

        val(val_data_cv, bigram_val_data_cv, model)


model = ClassifyMovie(len(word_to_ix), EMBEDDING_DIM, 2)
# model.load_embed(word_to_ix)
model.cuda()

for epoch in range(10):

    total_loss = train(train_data_cv, bigram_train_data_cv,  model)
    print("\nepoch " + str(epoch) + " " + str(total_loss[0]))

    # model.load_state_dict(torch.load('model/bigram_1'))

    val(val_data_cv, bigram_val_data_cv, model)

# torch.save(model.state_dict(), 'model/bigram_1')



# wv_vocab = model.load_embed()


