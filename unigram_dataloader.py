import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np

class SNLIDataset(Dataset):
    def __init__(self, transform=None):
        self.dataset = {'train':[], 'test':[], 'dev':[]}
        self.transform = transform
        self.word_to_idx = {}
        self.max_len = 0
        self.read_file()

    def read_file(self):
        filenames = ['train', 'dev', 'test' ]
        max_list = []
        # filenames = ['dev', 'test' ]
        labelDict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':0}
        dataset = {'train':[], 'test':[], 'dev':[]}
        vocab_set = set()
        for filename in filenames:
            print ('preprossing ' + filename + '...')
            fpr = open('data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
            fpr.readline()
            for line in fpr:
                sentences = line.strip().split('\t')
                tokens = [token for token in sentences[1].split(' ') if token != '(' and token != ')']
                tokens += [token for token in sentences[2].split(' ') if token != '(' and token != ')' ]
                vocab_set.update(tokens)
                
                dataset[filename].append((tokens, labelDict[sentences[0]]))
            fpr.close()
            self.word_to_idx = {word:i for i, word in enumerate(vocab_set, 1)}
            self.word_to_idx["[<pad>]"] = 0
            max_list += [max(map(self.len_of_sentence, dataset[filename]))]
            
        print ('SNLI preprossing finished!')
        self.max_len = max(max_list)
        # print(self.max_len)
        for filename in filenames:
            self.dataset[filename] = self.convert_data_to_word_to_idx(dataset[filename])
            print(len(self.dataset[filename]))
        

    def convert_data_to_word_to_idx(self, data):
        data_to_word_to_idx = []
        for sentence, label in data:
            sentence = [self.word_to_idx[w] for w in sentence]
            sentence_pad = np.zeros(self.max_len)
            sentence_pad[:len(sentence)] = sentence
            data_to_word_to_idx.append((np.array(sentence_pad, dtype=np.int64), np.array([label], dtype=np.int64)))
        return data_to_word_to_idx

    def len_of_sentence(self, input_tuple):
        sentence, _ = input_tuple
        return len(sentence)

    def __len__(self):
        return(len(self.dataset['train']))

    def __getitem__(self, idx):
        sample = {'sentence': self.dataset['train'][idx][0], 'label':self.dataset['train'][idx][1]}
        if self.transform:
            sample = self.transform(sample)
        # print('sample')
        # print(sample)
        return {'sentence': self.dataset['train'][idx][0], 'label':self.dataset['train'][idx][1]}

class ToTensor(object):
    """Covert ndarray in sample to Tensors."""
    def __call__(self, sample):
        sentence, label = sample['sentence'], sample['label']
        return {"sentence" : torch.from_numpy(sentence),
                'label': torch.from_numpy(label)}


dataset = SNLIDataset(transform=transforms.Compose([ToTensor()]))

# for i in range(2):
#     sample = dataset[i]
#     print(sample)

dataloader = DataLoader(dataset, batch_size=4, num_workers=4)

print("dataloader size")
print(len(dataloader))

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['sentence'],
          sample_batched['label'].shape)
    if (i_batch == 5):
        break