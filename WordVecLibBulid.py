from gensim.models import Word2Vec
from DataPreProcessing import dataPreProcess

source_data = dataPreProcess("source-data.csv")
texts = [data['text'] for data in source_data]
model_input_data = []
for sen in texts:
    model_input_data.append(sen.split(" ")[:-1])
print(model_input_data)
model = Word2Vec(model_input_data)
model.save("./word2vec/word2vec.model")
