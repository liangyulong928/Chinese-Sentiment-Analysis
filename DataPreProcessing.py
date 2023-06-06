import jieba
import csv
import re


def segDepart(sentence, stopwords):
    sentence_depart = jieba.cut(sentence.strip())
    outstrip = ''
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstrip += word
                outstrip += " "
    return outstrip


def splitTextAndLabel(data):
    sentences = []
    labels = []
    for sen in data:
        temp = sen[1].split(' __eou__ ')
        if len(temp) == len(sen[2]):
            for t in temp:
                sentences.append(t)
            for n in sen[2]:
                labels.append(n)
    return sentences, labels


def dataPreProcess(FileName):
    FilePath = "./data/"
    data = []
    with open(FilePath + FileName, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            data.append(row)
    data.pop(0)

    sentences, labels = splitTextAndLabel(data)

    stopwords = [line.strip() for line in open('./data/stopwords.txt', encoding='UTF-8').readlines()]

    seg_depart_sentences = []
    for sentence in sentences:
        seg_depart_sentences.append(segDepart(sentence, stopwords))

    clean_data = []
    for seg_depart_words, label in zip(seg_depart_sentences, labels):
        msg = {'text': re.sub(' +', ' ', seg_depart_words), 'label': int(label)-1}
        clean_data.append(msg)

    return clean_data