import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import nltk
import codecs
import numpy as np
import random

import pickle

from unigram_model import ClassifySentence
from unigram_load_data import prepSNLI
from unigram_dataloader import SNLIDataset, ToTensor

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

EMBEDDING_DIM = 100

_, test, dev, vocab = prepSNLI()
# print(len(train))

# random.shuffle(train)

print(len(vocab))
word_to_ix = {word: i for i, word in enumerate(vocab)}

loss_fucntion = nn.NLLLoss()

def train_model(model):
    model.train()
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    total_loss = torch.cuda.FloatTensor([0])
    # print(model.embeddings.weight.data[0].view(1,-1))
    for data in dataloader:
        sentence = Variable(data['sentence'].cuda())
        label = Variable(data['label'].cuda())

        # print("sentence ", sentence, label)
        offsets = Variable(torch.cuda.LongTensor([0]))
        model.zero_grad()

        log_probs = model(sentence, offsets)
        print(log_probs)
        print(label)
        loss = loss_fucntion(log_probs, label.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += (loss.data/len(dataloader))
    losses.append(total_loss.cpu().numpy()[0])
    return total_loss


def val_model(val_data, model):
    print("***Validation***")

    correct_count = 0
    val_loss = torch.cuda.FloatTensor([0])

    model.eval()
    for sentence, label in val_data:
        # print(sentence, label)

        sentence_idxs = [word_to_ix[w] for w in sentence]
        sentence_var = autograd.Variable(torch.cuda.LongTensor(sentence_idxs))

        offsets = Variable(torch.cuda.LongTensor([0]))

        log_probs = model(sentence_var, offsets)
        # print(log_probs)

        loss = loss_fucntion(log_probs, autograd.Variable(torch.cuda.LongTensor([label])))

        # print(log_probs)
        _, idx = torch.max(log_probs, 1)
        # print(idx)
        # print(label)
        val_loss += (loss.data/len(val_data))

        if idx.data.cpu().numpy() == label:
            correct_count += 1

    print("Accuracy  = ", correct_count*100./len(val_data))
    print("Accuracy(nllloss)  = ",val_loss[0])


print("instance : ", "1")
dataset = {x : SNLIDataset(x, transform=transforms.Compose([ToTensor()]))
            for x in ['train', 'test', 'dev']}

dataloader = {x : DataLoader(dataset[x], batch_size=4, num_workers=4)
                for x in ['train', 'test', 'dev']}

model = ClassifySentence(dataset['train'].len_vocab(), EMBEDDING_DIM, 3)

# model.load_embed(word_to_ix)
model.cuda()

for epoch in range(10):

    total_loss = train_model(dataloader['train'], model)
    torch.save(model.state_dict(), 'model/snli_data_fd')

    # model.load_state_dict(torch.load('model/bigram_1'))
    print("\nepoch " + str(epoch) + " " + str(total_loss[0]))

    val_model(dataloader['dev'], model)

# wv_vocab = model.load_embed()


