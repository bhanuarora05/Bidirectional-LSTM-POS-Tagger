import sys
import numpy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam

TRAIN_TIME_MINUTES = 11


class DatasetReader(object):

    @staticmethod
    def ReadFile(filename, term_index, tag_index):
        arr1 = []
        fin = open(filename,'r',encoding="utf8")
        for data in fin:
            data = data.strip()
            arr2 = []
            for sample in data.split():
                sample = sample.rsplit('/',1)
                emission = sample[0]
                posTag = sample[1]
                if emission in term_index:
                    pass
                else:
                    term_index[emission] = len(term_index)
                if posTag in tag_index:
                    pass
                else:
                    tag_index[posTag] = len(tag_index)
                arr2.append((term_index[emission],tag_index[posTag]))
            arr1.append(arr2)
        fin.close()
        return arr1

    @staticmethod
    def BuildMatrices(dataset):
        lngth = []
        for arr2 in dataset:
            lngth.append(len(arr2))
        max_length = max(lngth)
        lngth = numpy.array(lngth).astype(numpy.int64)
        trm_mtx = numpy.zeros((len(dataset), max_length)).astype(numpy.int64)
        for i in range(0,len(dataset)):
            for j in range(0,len(dataset[i])):
                trm_mtx[i,j] = dataset[i][j][0]
        tg_mtx = numpy.zeros_like(trm_mtx)
        for i in range(0,len(dataset)):
            for j in range(0,len(dataset[i])):
                tg_mtx[i,j] = dataset[i][j][1]
        return trm_mtx,tg_mtx,lngth


    @staticmethod
    def ReadData(train_filename, test_filename=None):
        term_index = {}
        tag_index = {}

        train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
        train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)

        if test_filename:
            test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
            test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

            if test_tags.shape[1] < train_tags.shape[1]:
                diff = train_tags.shape[1] - test_tags.shape[1]
                zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
                test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
                test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
            elif test_tags.shape[1] > train_tags.shape[1]:
                diff = test_tags.shape[1] - train_tags.shape[1]
                zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
                train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
                train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

            return (term_index, tag_index,
                    (train_terms, train_tags, train_lengths),
                    (test_terms, test_tags, test_lengths))
        else:
            return term_index, tag_index, (train_terms, train_tags, train_lengths)

class BiLSTM():
    EMBED_DIMEN = 256
    RNN_DIMEN = 512

    def __init__(self, max_length, num_terms, num_tags):
        self.lstm = Sequential()
        self.lstm.add(InputLayer(input_shape=(max_length,)))
        self.lstm.add(Embedding(num_terms,self.EMBED_DIMEN , mask_zero=True))
        self.lstm.add(Bidirectional(LSTM(self.RNN_DIMEN, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
        self.lstm.add(TimeDistributed(Dense(num_tags)))
        self.lstm.add(Activation('softmax'))
        self.lstm.compile(loss='categorical_crossentropy',optimizer=Adam(0.005, decay=0.001),metrics=['accuracy'])
    
    def LSTM(self):
        return self.lstm

class SequenceModel(object):
    BATCH_SIZE = 128

    def __init__(self, max_length=310, num_terms=1000, num_tags=40):
        self.max_length = max_length
        self.num_terms = num_terms
        self.num_tags = num_tags

    def lengths_vector_to_binary_matrix(self, length_vector):
        return tf.sequence_mask(length_vector, self.max_length, dtype=tf.float32)

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass

    def build_inference(self):
        pass

    def run_inference(self, tags, lengths):
        temp = numpy.copy(tags)
        for i in range(len(lengths)): temp[i, :lengths[i]] += 1
        logits = self.lstm.LSTM().predict(temp,len(temp))
        return numpy.argmax(logits, axis=2)

    def build_training(self):
        self.lstm = BiLSTM(self.max_length,self.num_terms+1,self.num_tags)

    def train_epoch(self, terms, tags, lengths, batch_size=32, learn_rate=1e-7):
        temp = numpy.copy(terms)
        for i in range(len(lengths)): temp[i, :lengths[i]] += 1
        tags = (numpy.arange(self.num_tags) == tags[...,None]).astype(float)
        self.lstm.LSTM().fit(temp, tags, batch_size=self.BATCH_SIZE, epochs=1, validation_split=0.0)
        return True

    def evaluate(self, terms, tags, lengths):
        predicted_tags = self.run_inference(terms, lengths)
        if predicted_tags is None:
          print('Is your run_inference function implented?')
          return 0
        test_accuracy = numpy.sum(
          numpy.cumsum(numpy.equal(tags, predicted_tags), axis=1)[numpy.arange(lengths.shape[0]),lengths-1])/numpy.sum(lengths + 0.0)
        return test_accuracy

def main():
    reader = DatasetReader
    train_filename = sys.argv[1]
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data

    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.build_training()
    for j in range(5):
        model.train_epoch(train_terms, train_tags, train_lengths)
        print('Finished epoch %i. Evaluating ...' % (j+1))
        print (model.evaluate(test_terms, test_tags, test_lengths))


if __name__ == '__main__':
    main()