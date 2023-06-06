from sklearn.metrics import classification_report

from DataPreProcessing import dataPreProcess

emo_dict = {}
deg_dict = {}
with open('./data/dict/positive.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        emo_dict[line] = 1
with open('./data/dict/negative.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        emo_dict[line] = -1
with open('./data/dict/ish.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        deg_dict[line] = -0.5
with open('./data/dict/inverse.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        deg_dict[line] = -1
with open('./data/dict/more.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        emo_dict[line] = 1.5
with open('./data/dict/very.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        emo_dict[line] = 2
with open('./data/dict/over.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        emo_dict[line] = 2.5
with open('./data/dict/most.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        emo_dict[line] = 3


if __name__ == '__main__':
    source_data = dataPreProcess("source-data.csv")
    texts = [data['text'] for data in source_data]
    labels = [1 if data['label'] < 2 else 0 if data['label'] == 5 else -1 for data in source_data]
    scores = []
    num = 0
    for sen in texts:
        score = 0
        temp = 0
        for word in sen:
            if word in emo_dict:
                if temp:
                    if temp * emo_dict[word] > 0:
                        score = score + temp * emo_dict[word]
                    else:
                        score = score + temp * emo_dict[word]
                    temp = 0
                else:
                    score += emo_dict[word]
                num += 1
            elif word in deg_dict:
                if temp:
                    temp *= deg_dict[word]
                else:
                    temp = deg_dict[word]
                num += 1
        if temp != 0:
            if temp * score > 0:
                score = score + temp
            elif temp * score < 0:
                score = 0 - score + temp
            else:
                score = temp
        scores.append(1 if score > 0 else 0 if score == 0 else -1)
    finalScore = classification_report(labels, scores)
    print(finalScore)
