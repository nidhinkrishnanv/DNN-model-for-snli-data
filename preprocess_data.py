import gensim
import os
import time
import glob
import sys
import pickle
import re
import numpy as np

import torch
import torch.nn as nn
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

def vocab():
    filenames = ['dev', 'test', 'train']
    # filenames = ['test', 'dev']
    labelDict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':0}
    vocab = set()
    max_len = 0
    for filename in filenames:
        print ('preprossing ' + filename + '...')
        fpr = open('../data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
        count = 0
        fpr.readline()
        for line in fpr:
            # if count > 1:
            #     break
            sentences = line.strip().split('\t')
            sentences = line.strip().split('\t')
            tokens = [token for token in sentences[1].split(' ') if token != '(' and token != ')' and token not in stopWords]
            tokens += [token for token in sentences[2].split(' ') if token != '(' and token != ')' and token not in stopWords]
            max_len = max([max_len, len(tokens)])
            vocab.update(tokens)
            count += 1   
        fpr.close()
    # print(vocab)
    word_to_idx = {word:i for i, word in enumerate(vocab, 1)}
    word_to_idx["[<pad>]"] = 0
    print("Vocab size : ", len(word_to_idx))
    print("max_len " ,max_len)
    print ('SNLI preprossing finished!')

    with open('data/word_to_idx.json', 'w') as f:
        json.dump(word_to_idx, f)

    with open('data/max_len.pkl', 'wb') as f:
        pickle.dump(max_len, f)

    
def get_word_to_ix():
    with open('data/word_to_idx.json') as f:
        return json.load(f)

def get_max_len():
    with open('data/max_len.pkl', 'rb') as f:
        return pickle.load(f)



def save_data(is_train_gensim=False):
    filenames = ['dev', 'test', 'train']
    labelDict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':0}
    vocab = set()
    gensim_train_data = []
    for filename in filenames:
        data = []
        print ('preprossing ' + filename + '...')
        fpr = open('../data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
        fpr.readline()
        count = 0
        for line in fpr:
            # if count >= 1000:
            #     break
            sentences = line.strip().split('\t')
            tokens = [[token for token in sentences[x].split(' ') if token != '(' and token != ')' and token not in stopWords] for x in [1, 2]]
            
            #For training gensim model
            if is_train_gensim:
                gensim_train_data.extend([tokens[0], tokens[1]])

            vocab.update(tokens[0]); vocab.update(tokens[1])

            #check for empty strings
            if len(tokens[0]) != 0 and len(tokens[1]) != 0:
                data.append(([tokens[0], tokens[1]], labelDict[sentences[0]]))
                # print(count)
                # print(tokens[0])
                # print(tokens[1])
                # print(sentences[1])
                # print(sentences[2])
            count += 1
        fpr.close()
        # print(data[:4])
        print ('SNLI preprossing ' + filename + ' finished!\n')
        print("Saving Data Vocab size : ", len(vocab))
        data = convert_data_to_word_to_idx(data)
        with open('data/' + filename + '.pkl', 'wb') as f:
            pickle.dump(data, f)
    if is_train_gensim:
        model = gensim.models.Word2Vec(gensim_train_data, size=300, window=5, min_count=1, workers=4)
        model.save('model/word2vec_snli.model')

def convert_data_to_word_to_idx(data):
    data_to_word_to_idx = []
    word_to_idx = get_word_to_ix()
    for sentence, label in data:
        sent_w2idx = [[word_to_idx[w] for w in sentence[i]]  for i in range(2)]
        data_to_word_to_idx.append((np.array(sent_w2idx[0], dtype=np.int64), 
            np.array(sent_w2idx[1], dtype=np.int64),
            np.array([label], dtype=np.int64)))
    # print(data_to_word_to_idx[0])
    return data_to_word_to_idx

def load_data(filename):
    with open('data/' + filename + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    print(stopWords)
    #save word_to_idx
    # vocab()
    save_data()
    #gensim
    # w2v = TrainWord2Vec()
    # w2v.train()
    # model = gensim.models.Word2Vec.load('model/word2vec_snli.model')
    # print(type(model))
    # print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
