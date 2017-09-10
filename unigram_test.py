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



EMBEDDING_DIM = 300

# dataset = SNLIDataset('test', transform=transforms.Compose([ToTensor()]))
# dset_loader = DataLoader(dataset['test'], batch_size=512, shuffle=True, num_workers=4, collate_fn=packed_collate_fn)
# dset_sizes = len(dataset['test'])
# print(dset_sizes)


def test_model(model, dset_loader, dset_size, loss_function, lr_scheduler=None):
    since = time.time()

    if lr_scheduler:
        optimizer = lr_scheduler(optimizer, epoch)

    model.eval()
    running_loss = 0.0
    running_corrects = 0

    # Iterate ove data
    for data in dset_loader:
        #get the inputs
        sentences = [Variable(data['sent'][i].cuda()) for i in range(2)]
        sent_lengths = [Variable(data['sent_length'][i].float().cuda()) for i in range(2)]
        labels = Variable(data['labels'].cuda())
        # print(model.embeddings.weight.data[0])

        outputs = model(sentences, sent_lengths)
        _, preds = torch.max(outputs.data, 1)
        loss = loss_function(outputs, labels.view(-1))

        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data.view(-1))

            # print(running_corrects)

    loss = running_loss / dset_size
    acc = running_corrects / dset_size

    print('Loss: {:.4f} Acc: {:.4f}'.format(
        loss, acc))

    print()

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0001, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

