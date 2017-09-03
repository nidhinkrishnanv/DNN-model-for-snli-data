import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchwordemb
import gensim
import numpy as np

class ClassifyMovie(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(ClassifyMovie, self).__init__()

        self.embed_unigram = nn.EmbeddingBag(vocab_size, embedding_dim)
        
        self.l1 = nn.Dropout(p=0.4)

        self.hidden_dim = 100
        self.hidden = self.init_hidden()
        self.embed_bigram = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)

        self.l2 = nn.Linear(embedding_dim + self.hidden_dim, num_classes)

    def forward(self, inputs_unigram, input_bigram, offset):

        out_unigram = self.embed_unigram(inputs_unigram, offset)
        
        # out_unigram = self.l1(out_unigram)
        
        out_bigram = self.embed_bigram(input_bigram)
        out_bigram = torch.sum(out_bigram, dim=1)
        lstm_out, self.hidden = self.lstm(out_bigram.view(len(input_bigram),1,-1), self.hidden)

        # print(out_unigram.data.shape)
        # print(self.hidden[0].view(-1, self.hidden_dim).data.shape)
        final = torch.cat([out_unigram, self.hidden[0].view(-1, self.hidden_dim)], dim=1)

        final = self.l1(final)


        out = F.log_softmax(self.l2(final))
        return out

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
            autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))

    def load_embed(self, word_to_ix):
        #Small size | EMBEDDING_DIM = 50
        wv_vocab, vec = torchwordemb.load_glove_text("model/glove.6B.50d.txt")
        #Large size | EMBEDDING_DIM = 300
        # wv_vocab, vec = torchwordemb.load_glove_text("model/glove.6B.100d.txt")
        # wv_vocab, vec = torchwordemb.load_word2vec_bin("model/GoogleNews-vectors-negative300.bin")
        self.embed_bigram.weight = nn.Parameter(vec)

        new_weights_t = torch.zeros(len(word_to_ix), vec.shape[1])

        for word in word_to_ix:
            # print(word)
            new_weights_t[word_to_ix[word]] = vec[wv_vocab[word]]

        # print('apple')
        # print(print(vec[ wv_vocab["apple"] ] ))
        # print(new_weights_t[word_to_ix['apple']])

        self.embed_bigram.weight = nn.Parameter(new_weights_t)
        # self.embeddings.weight.requires_grad = False

        return wv_vocab


# CM = ClassifyMovie(17634, 50)
# CM.load_embed()
