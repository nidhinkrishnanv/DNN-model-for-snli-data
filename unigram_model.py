import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchwordemb
import gensim
import numpy as np

class ClassifySentence(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_classes, dropout):
        super(ClassifySentence, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.l1 = nn.Dropout(p=dropout)
        self.l2 = nn.Linear(embedding_dim, num_classes)
        # self.set_padding_embed()

    def forward(self, inputs):
        # print(inputs)
        out = self.embeddings(inputs)
        # print('embeddings ', out.data.shape)
        out = torch.sum(out, dim=1)
        out = F.log_softmax(self.l2(out))
        return out

    def set_padding_embed(self):
        self.embeddings.weight.data[0].zero_()






    def load_embed(self, word_to_ix):
        #Small size | EMBEDDING_DIM = 50
        wv_vocab, vec = torchwordemb.load_glove_text("model/glove.6B.50d.txt")
        #Large size | EMBEDDING_DIM = 300
        # wv_vocab, vec = torchwordemb.load_glove_text("model/glove.6B.100d.txt")
        # wv_vocab, vec = torchwordemb.load_word2vec_bin("model/GoogleNews-vectors-negative300.bin")
        self.embeddings.weight = nn.Parameter(vec)

        new_weights_t = torch.zeros(len(word_to_ix), vec.shape[1])

        for word in word_to_ix:
            # print(word)
            new_weights_t[word_to_ix[word]] = vec[wv_vocab[word]]

        # print('apple')
        # print(print(vec[ wv_vocab["apple"] ] ))
        # print(new_weights_t[word_to_ix['apple']])

        self.embeddings.weight = nn.Parameter(new_weights_t)
        # self.embeddings.weight.requires_grad = False

        return wv_vocab


# CM = ClassifySentence(17634, 50,2)
# CM.load_embed()
