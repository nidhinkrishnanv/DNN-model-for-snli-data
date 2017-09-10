import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
import copy

import pickle

from unigram_model import ClassifySentence
from snli_dataset import SNLIDataset, ToTensor, packed_collate_fn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from unigram_test import test_model

EPOCHS = 30

EMBEDDING_DIM = 300

dataset = {x : SNLIDataset(x, transform=transforms.Compose([ToTensor()]))
            for x in ['train', 'test', 'dev']}
dset_loader = {x : DataLoader(dataset[x], batch_size=512, shuffle=True, num_workers=4, collate_fn=packed_collate_fn)
                for x in ['train', 'test', 'dev']}
dset_sizes = {x : len(dataset[x]) for x in ['train', 'test', 'dev'] }
print(dset_sizes)


def train_model(model, loss_function, optimizer, lr_scheduler=None, num_epochs=5):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                if lr_scheduler:
                    optimizer = lr_scheduler(optimizer, epoch)
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate ove data
            for data in dset_loader[phase]:
                #get the inputs
                sentences = [Variable(data['sent'][i].cuda()) for i in range(2)]
                sent_lengths = [Variable(data['sent_length'][i].float().cuda()) for i in range(2)]
                labels = Variable(data['labels'].cuda())
                # print(model.embeddings.weight.data[0])

                optimizer.zero_grad()

                outputs = model(sentences, sent_lengths)
                _, preds = torch.max(outputs.data, 1)
                loss = loss_function(outputs, labels.view(-1))

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # print(preds)
                # print(labels.data)
                # print(preds == labels.data)
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data.view(-1))

                # print(running_corrects)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            #deep copy the model
            if phase == 'dev' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model, best_acc

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0001, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def hyperparm_tune():
    best_acc = 0
    best_w = 0
    best_d = 0
    for i, (w, d) in enumerate(zip(layer_width, dropout)):
        print()
        print('{} Weight: {} Dropout: {}'.format(i, w, d))
        model = ClassifySentence(dataset['train'].len_vocab(), EMBEDDING_DIM, 3, d, w)
        model.cuda()        
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr=0.001)
        model_ft, acc = train_model(model, loss_function, optimizer, num_epochs=EPOCHS)

        # model_ft, acc = train_model(model, loss_function, optimizer, exp_lr_scheduler, 25)
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_d = d
    print('best_w: {:.4f} best_d: {:.4f} best_acc: {:.4f}'.format(
        best_w, best_d, best_acc))


if __name__ == "__main__":


    loss_function = nn.NLLLoss()

    # weight = 10**(-7*np.random.rand(15))
    # dropout = np.random.rand(15)
    # layer_width = [500, 600, 700, 800, 900]*3
    # # com = [(x,y) for x in weight for y in dropout]
    # print(dropout)
    # print(layer_width)
    # hyperparm_tune()

    model = ClassifySentence(dataset['train'].len_vocab(), EMBEDDING_DIM, 3, 0.2, 700)
    model.cuda()

    temp = model.embeddings.weight[3].data.cpu().numpy()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.Adam(parameters, lr=0.001, weight_decay=1.287051167874816e-06)
    optimizer = optim.Adam(parameters, lr=0.001)
    # model_ft, acc = train_model(model, loss_function, optimizer, exp_lr_scheduler, 30)
    model_ft, acc = train_model(model, loss_function, optimizer, num_epochs=EPOCHS)

    #test model
    print("\nTesting")
    test_model(model_ft, dset_loader['test'], dset_sizes['test'], loss_function)


# print(np.array_equal(temp, model.embeddings.weight[3].data.cpu().numpy()))

