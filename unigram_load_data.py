import os
import time
import glob
import sys

import torch
import torch.optim as O
import torch.nn as nn


def prepSNLI():
    filenames = ['dev', 'test', 'train']
    filenames = ['test', 'dev']
    labelDict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':0}
    train = []
    test = []
    dev = []
    # bigram_train = []
    # bigram_test = []
    vocab = set()
    for filename in filenames:
        print ('preprossing ' + filename + '...')
        fpr = open('data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
        count = 0
        fpr.readline()
        for line in fpr:
            sentences = line.strip().split('\t')
            # if sentences[0] == '-':
            #     continue

            tokens1 = sentences[1].split(' ')
            tokens1 = [token for token in tokens1 if token != '(' and token != ')']

            vocab.update(tokens1)

            tokens2 = sentences[2].split(' ')
            tokens2 = [token for token in tokens2 if token != '(' and token != ')' ]
            
            vocab.update(tokens2)

            tokens1.extend(tokens2)


            if filename == 'train':
                train.append((tokens1, labelDict[sentences[0]]))

            if filename == 'test':
                test.append((tokens1, labelDict[sentences[0]]))

            if filename == 'dev':
                dev.append((tokens1, labelDict[sentences[0]]))
            count += 1
        
        fpr.close()
    print ('SNLI preprossing finished!')
    return train, test, dev, vocab
