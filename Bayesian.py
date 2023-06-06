import torch
from DataForTfIdf import dataLoader
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from DataPreProcessing import dataPreProcess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    source_data = dataPreProcess("source-data.csv")
    train_data, train_labels, text_data, text_labels, index = dataLoader(source_data)
    print("加载模型")
    model = MultinomialNB()
    model.fit(train_data, train_labels)
    print("模型训练完成")
    predict_result_for_Bayesian = []
    for text_sen in text_data:
        predict_result_for_Bayesian.append(model.predict(text_sen.reshape(1, -1))[0])
    score = classification_report(text_labels, predict_result_for_Bayesian)
    print(score)
