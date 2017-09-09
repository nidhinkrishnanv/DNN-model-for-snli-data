import gensim

class TrainWord2Vec():
    def __init__(self):
        self.data = []
        self.read_file()

    def read_file(self):
        data = []
        filenames = ['dev', 'test', 'train']
        # model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)
        for filename in filenames:
            print ('preprossing ' + filename + '...')
            fpr = open('data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
            fpr.readline()
            count = 0
            for line in fpr:
                # if count > 3:
                #     break
                sentences = line.strip().split('\t')
                tokens = [[token for token in sentences[x].split(' ') if token != '(' and token != ')'] for x in [1, 2]]
                data.extend([tokens[0], tokens[1]])
                count += 1
            fpr.close()
            # print(data)

        model = gensim.models.Word2Vec(data, size=300, window=5, min_count=1, workers=4)
        # model.train(data)
        model.save('model/word2vec_snli.model')
            # self.data = self.convert_data_to_word_to_idx(data)
        

    def convert_data_to_word_to_idx(self, data):
        data_to_word_to_idx = []

        for sentences, score in data:
            # sentences_pad = [np.zeros(self.max_len) for i in range(2)]
            for i in range(2):
                sentences[i] = [self.word_to_idx[w] for w in sentences[i]]
                # sentences_pad[i][:len(sentences[i])] = sentences[i]

            data_to_word_to_idx.append((np.array(sentences[0], dtype=np.int64),
                np.array(sentences[1], dtype=np.int64),
                # np.array([len(sentences[0])], dtype=np.int64),
                # np.array([len(sentences[1])], dtype=np.int64),
                np.array([score], dtype=np.float64)))

        # print('data_to_word_to_idx', data_to_word_to_idx[0])
        return data_to_word_to_idx

if __name__ == "__main__":
   # tW2V = TrainWord2Vec()
   # model = gensim.models.Word2Vec.load('model/word2vec_snli.model')
   # print(type(model))
   # print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
