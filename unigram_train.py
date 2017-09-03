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
import time

import pickle

from unigram_model import ClassifySentence
from unigram_load_data import prepSNLI
from unigram_dataloader import SNLIDataset, ToTensor

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

EMBEDDING_DIM = 100

dataset = {x : SNLIDataset(x, transform=transforms.Compose([ToTensor()]))
            for x in ['train', 'test', 'dev']}
dset_loader = {x : DataLoader(dataset[x], batch_size=4, num_workers=4)
                for x in ['train', 'test', 'dev']}
dset_sizes = {x : len(dataset[x]) for x in ['train', 'test', 'dev'] }

model = ClassifySentence(dataset['train'].len_vocab(), EMBEDDING_DIM, 3)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_function = nn.NLLLoss()
model.cuda()

def train_model(model, loss_function, optimizer, lr_scheduler=None, num_epochs=5):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}', format(epoch, num_epochs - 1))
        print('-' * 10)

        #Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase = 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate ove data
            for data in dset_loader[phase]:
                #get the inputs
                sentence = Variable(data['sentence'].cuda())
                label = Variable(data['label'].cuda())

                optimizer.zero_grad()

                outputs = model(sentence)
                _, preds = troch.max(outputs.data, 1)
                loss = loss_function(outputs, label.view(-1))

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}', format(
                phase, epoch_loss. epoch_acc))

            #deep copy the model
            if phase == 'val' and epoc_acc > best_acc:
                best_acc = epoc_acc
                best_model = copy.deepcopy(mode)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s',format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

model_ft = train_model(model, loss_function, optimizer)


