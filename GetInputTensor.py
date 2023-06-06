import torch
from gensim.models import Word2Vec

from DataForTfIdf import dataLoader
from DataPreProcessing import dataPreProcess


word2vecModel = Word2Vec.load("./word2vec/word2vec.model")


def buildNNInput(data, index, name):
    train_data_list = []
    for tf_idf in data:
        tf_idf = tf_idf.tolist()
        num = 0
        vec_list = []
        while num < 4:
            if max(tf_idf) != 0:
                try:
                    vec_list.append(list(word2vecModel.wv[index[tf_idf.index(max(tf_idf))]]))
                    tf_idf[tf_idf.index(max(tf_idf))] = 0
                    num += 1
                except:
                    tf_idf[tf_idf.index(max(tf_idf))] = 0
            else:
                vec_list.append([0] * 100)
                num += 1
        train_data_list.append(vec_list)
    torch.save(torch.tensor(train_data_list), name)


if __name__ == '__main__':
    source_data = dataPreProcess("source-data.csv")
    train_data, train_labels, text_data, text_labels, index = dataLoader(source_data)
    buildNNInput(train_data, index, "./tensor/train_data_tensors.pt")
    buildNNInput(text_data, index, "./tensor/text_data_tensors.pt")
