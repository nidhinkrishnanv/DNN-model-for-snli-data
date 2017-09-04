import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from create_word_to_ix import get_word_to_ix, get_max_len

import numpy as np

class SNLIDataset(Dataset):
    def __init__(self, filename, transform=None):
        self.data = []
        self.transform = transform
        self.word_to_idx = get_word_to_ix()
        self.max_len = get_max_len()
        self.read_file(filename)

    def read_file(self, filename):
        max_list = []
        labelDict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':0}
        # dataset = {'train':[], 'test':[], 'dev':[]}
        data = []

        print ('preprossing ' + filename + '...')
        fpr = open('data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
        fpr.readline()
        count = 0
        for line in fpr:
            # if count >= 4:
            #     break
            count += 1
            sentences = line.strip().split('\t')
            tokens = [token for token in sentences[1].split(' ') if token != '(' and token != ')']
            tokens += [token for token in sentences[2].split(' ') if token != '(' and token != ')' ]
            data.append((tokens, labelDict[sentences[0]]))
        fpr.close()
        # print ('SNLI preprossing ' + filename + ' finished!')
        # print("Vocab size : ", len(self.word_to_idx))
        self.data = self.convert_data_to_word_to_idx(data)
        

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
        return(len(self.data))

    def len_vocab(self):
        return len(self.word_to_idx)

    def __getitem__(self, idx):
        sample = {'sentence': self.data[idx][0], 'label':self.data[idx][1]}
        if self.transform:
            sample = self.transform(sample)
        # print('sample')
        # print(sample)
        return {'sentence': self.data[idx][0], 'label':self.data[idx][1]}

class ToTensor(object):
    """Covert ndarray in sample to Tensors."""
    def __call__(self, sample):
        sentence, label = sample['sentence'], sample['label']
        return {"sentence" : torch.from_numpy(sentence),
                'label': torch.from_numpy(label)}



if __name__ == "__main__":
    dataset = SNLIDataset('test', transform=transforms.Compose([ToTensor()]))

    for i in range(2):
        sample = dataset[i]
        print(sample)

    dataloader = DataLoader(dataset, batch_size=4, num_workers=4)

    # print("dataloader size")
    # print(len(dataloader))


    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['sentence'].size(),
              sample_batched['label'].shape)
        if (i_batch == 5):
            break