import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchwordemb
import gensim
import numpy as np
from preprocess_data import get_word_to_ix, get_max_len

class ClassifySentence(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_classes, dropout, layer_width=700):
        super(ClassifySentence, self).__init__()
        self.embedding_dim = embedding_dim
        self.layer_width = layer_width
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.l1 = nn.Linear(2*embedding_dim, self.layer_width)
        self.d1 = nn.Dropout(p=dropout)
        self.l2 = nn.Linear(self.layer_width, self.layer_width)
        self.d2 = nn.Dropout(p=dropout)
        self.l3 = nn.Linear(self.layer_width, num_classes)
        # self.set_padding_embed()
        # print(self.embeddings.weight[3])
        self.load_embed()

        # print(self.embeddings.weight[3])

    def forward(self, sentences, sent_lengths):
        # print(inputs)
        out = [self.embeddings(sentences[i]) for i in range(2)]
        out = [torch.sum(out[i], dim=1)/sent_lengths[i].view(-1, 1) for i in range(2)]
        # print('embed', out[0].data.shape)
        out = torch.cat([out[0], out[1]],dim=1)
        # print('out s', out.data.shape)
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
