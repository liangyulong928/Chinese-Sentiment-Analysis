from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def dataLoader(source_data):
    # labels_for_6_classifier = [data['label'] for data in source_data]
    # labels_for_5_classifier = []
    # for data in source_data:
    #     if data['label'] != 5:
    #         labels_for_5_classifier.append(data['label'])
    texts = [data['text'] for data in source_data]
    labels = [1 if data['label'] < 2 else 0 if data['label'] == 5 else 2 for data in source_data]
    # labels = []
    # texts = []
    # for data in source_data:
    #     if data['label'] != 5:
    #         labels.append(1 if data['label'] < 2 else 0)
    #         # labels.append(data['label'])
    #         texts.append(data['text'][:-1])
    vectorised = TfidfVectorizer()
    texts_tf_idf = vectorised.fit_transform(texts).toarray()
    index = vectorised.get_feature_names_out()
    train_data = texts_tf_idf[:len(texts) // 9 * 8]
    text_data = texts_tf_idf[len(texts) // 9 * 8:]
    train_labels = np.array(labels[:len(texts) // 9 * 8])
    text_labels = labels[len(texts) // 9 * 8:]
    return train_data, train_labels, text_data, text_labels, index
