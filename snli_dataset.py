import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from create_word_to_ix import get_word_to_ix, get_max_len
from preprocess_data import load_data

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

import numpy as np


def packed_collate_fn(data, is_sort=False):

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, torch.LongTensor(lengths)

    enumerated_data = [[idx, x[0], x[1], x[2]] for idx, x in enumerate(data)]
    
    sent_order = [None]*2; sent_lengths = [None]*2 
    sent_seqs = [None]*2; merged_seq = [None]*2

    for i in range(2):
        if is_sort:
            #sort for sentence 1 and 2 and create batch
            enumerated_data.sort(key=lambda sent: len(sent[i+1]), reverse=True)
        sent_order[i], sent_seqs[0], sent_seqs[1], _ = zip(*enumerated_data)
        merged_seq[i], sent_lengths[i] = merge(sent_seqs[i])

    label = [x[2][0] for x in data]
    label = torch.LongTensor(label)

    return {'sent':merged_seq, 'sent_length':sent_lengths, 'order':sent_order, 'labels':label}

def packed_collate_sort(data, is_sort=True):
    return packed_collate_fn(data, is_sort)


class SNLIDataset(Dataset):
    def __init__(self, filename, transform=None):
        self.data = []
        self.transform = transform
        self.word_to_idx = get_word_to_ix()
        self.max_len = get_max_len()
        self.read_file(filename)

    def read_file(self, filename):
        self.data = load_data(filename)
        # print(data[:4])

    def len_of_sentence(self, input_tuple):
        sentence, _ = input_tuple
        return len(sentence)

    def __len__(self):
        return(len(self.data))

    def len_vocab(self):
        return len(self.word_to_idx)

    def __getitem__(self, idx):
        sample = {'sentence1': self.data[idx][0],
            'sentence2': self.data[idx][1],
            'label' : self.data[idx][2]}
        if self.transform:
            sample = self.transform(sample)
        # print(idx)
        return sample

class ToTensor(object):
    """Covert ndarray in sample to Tensors."""
    def __call__(self, sample):
        sentence1, sentence2, label = sample['sentence1'], sample['sentence2'], sample['label']
        
        # print(sentence1, sentence2, label)
        return [torch.from_numpy(sentence1), torch.from_numpy(sentence2), 
            # torch.from_numpy(len_sent1), torch.from_numpy(len_sent2), 
            torch.from_numpy(label)]



if __name__ == "__main__":
    dataset = SNLIDataset('test', transform=transforms.Compose([ToTensor()]))

    # for i in range(2):
    #     sample = dataset[i]
    #     print(sample)

    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, collate_fn=packed_collate_fn)

    # print("dataloader size")
    # print(len(dataloader))


    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched,)
        print(sample_batched['sent'][0])
        if (i_batch == 1):
            break

