import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from create_word_to_ix import get_word_to_ix, get_max_len

import numpy as np


def packed_collate_fn(data):

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    enumerated_data = [[idx, x[0], x[1], x[2]] for idx, x in enumerate(data)]
    
    #sort for sentence1 and create batch
    enumerated_data.sort(key=lambda sent: len(sent[1]), reverse=True)
    sent1_order, sent1_seqs, _, _ = zip(*enumerated_data)
    sent1_seqs, sent1_lengths = merge(sent1_seqs)

    #sort for sentence2 and create batch
    enumerated_data.sort(key=lambda sent: len(sent[2]), reverse=True)
    sent2_order, _, sent2_seqs, _ = zip(*enumerated_data)
    sent2_seqs, sent2_lengths = merge(sent2_seqs)

    score = [x[2].numpy() for x in data]
    score = torch.Tensor(score)

    return sent1_seqs, sent1_lengths, sent1_order, sent2_seqs, sent2_lengths, sent2_order, score


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
        fpr = open('../data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
        fpr.readline()
        count = 0
        for line in fpr:
            if count >= 100:
                break
            sentences = line.strip().split('\t')
            tokens = [[token for token in sentences[x].split(' ') if token != '(' and token != ')'] for x in [1, 2]]
            data.append(([tokens[0], tokens[1]], labelDict[sentences[0]]))
            count += 1
        fpr.close()
        # print ('SNLI preprossing ' + filename + ' finished!')
        # print("Vocab size : ", len(self.word_to_idx))
        self.data = self.convert_data_to_word_to_idx(data)
        

    def convert_data_to_word_to_idx(self, data):
        data_to_word_to_idx = []
        for sentence, label in data:
            for i in range(2):
                sentence[i] = [self.word_to_idx[w] for w in sentence]
            
            data_to_word_to_idx.append((np.array(sentence[0], dtype=np.int64), 
                np.array(sentence[1], dtype=np.int64),
                np.array([label], dtype=np.int64)))
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

