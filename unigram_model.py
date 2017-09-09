import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchwordemb
import gensim
import numpy as np
from create_word_to_ix import get_word_to_ix, get_max_len

class ClassifySentence(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_classes, dropout):
        super(ClassifySentence, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.l1 = nn.Linear(embedding_dim, 700)
        self.d1 = nn.Dropout(p=dropout)
        self.l2 = nn.Linear(700, 700)
        self.d2 = nn.Dropout(p=dropout)
        self.l3 = nn.Linear(700, num_classes)
        # self.set_padding_embed()
        # print(self.embeddings.weight[3])
        self.load_embed()

        # print(self.embeddings.weight[3])

    def forward(self, inputs):
        # print(inputs)
        out = self.embeddings(inputs)
        # print('embeddings ', out.data.shape)
        out = torch.sum(out, dim=1)
        out = F.relu(self.d1(self.l1(out)))
        out = F.relu(self.d2(self.l2(out)))
        out = F.log_softmax(self.l3(out))
        return out

    def set_padding_embed(self):
        self.embeddings.weight.data[0].zero_()

    def load_embed(self):
        word_to_ix = get_word_to_ix()
        #Small size | EMBEDDING_DIM = 50
        # wv_vocab, vec = torchwordemb.load_glove_text("model/glove.6B.50d.txt")
        #Large size | EMBEDDING_DIM = 100
        # wv_vocab, vec = torchwordemb.load_glove_text("model/glove.6B/glove.6B.300d.txt")
        # wv_vocab, vec = torchwordemb.load_word2vec_bin("model/GoogleNews-vectors-negative300.bin")
        model = gensim.models.Word2Vec.load('model/word2vec_snli.model')
        # print(len(model.wv.vocab))
        # print(len(model['apple']))
        # print(model.size)

        new_weights_t = torch.zeros(len(word_to_ix), self.embedding_dim)

        for word in word_to_ix:
            if word != '[<pad>]':
                new_weights_t[word_to_ix[word]] = torch.Tensor(model[word])

        # print('apple')
        # print(print(vec[ wv_vocab["apple"] ] ))
        # print(new_weights_t[word_to_ix['apple']])

        self.embeddings.weight = nn.Parameter(new_weights_t)
        
        self.embeddings.weight.requires_grad = False


# CM = ClassifySentence(17634, 50,2)
# CM.load_embed()
