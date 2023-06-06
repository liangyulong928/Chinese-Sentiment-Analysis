import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from DataForTfIdf import dataLoader
from DataPreProcessing import dataPreProcess
from resultCompare import ResultCompare


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 20, 1)
        self.conv2 = nn.Conv1d(4, 20, 2)
        self.conv3 = nn.Conv1d(4, 20, 4)
        self.relu = nn.ReLU()
        self.droput = nn.Dropout(0.3)
        self.linear = nn.Linear(60, 3)

    def forward(self, x):
        outputs = []
        output_kernel_2 = self.conv1(x)
        output_kernel_3 = self.conv2(x)
        output_kernel_4 = self.conv3(x)
        output_kernel_2 = torch.mean(output_kernel_2, dim=2)
        output_kernel_3 = torch.mean(output_kernel_3, dim=2)
        output_kernel_4 = torch.mean(output_kernel_4, dim=2)
        output_kernel_2 = self.droput(output_kernel_2)
        output_kernel_3 = self.droput(output_kernel_3)
        output_kernel_4 = self.droput(output_kernel_4)
        outputs.append(output_kernel_2)
        outputs.append(output_kernel_3)
        outputs.append(output_kernel_4)
        output_vector = torch.cat(outputs, dim=1)
        output_vector = self.linear(output_vector)
        return output_vector


# 创建模型实例
model = CNN()
batch_size = 16
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    train_tensor = torch.load("./tensor/train_data_tensors.pt")
    test_tensor = torch.load("./tensor/text_data_tensors.pt")
    source_data = dataPreProcess("source-data.csv")
    train_data, train_labels, text_data, text_labels, index = dataLoader(source_data)
    train_dataset = TensorDataset(train_tensor, torch.tensor(train_labels))
    text_dataset = TensorDataset(train_tensor, torch.tensor(train_labels))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for inputs, targets in train_dataloader:
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    text_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)
    outputs_list = []
    for inputs, targets in text_dataloader:
        model.eval()
        outputs = model(inputs)
        outputs = torch.argmax(outputs, dim=1).tolist()
        outputs_list.extend(outputs)
    ResultCompare(outputs_list, text_labels)
