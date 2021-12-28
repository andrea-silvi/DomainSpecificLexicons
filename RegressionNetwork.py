import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionModel(nn.Module):

    def __init__(self, low, high, dropout=0.2):
        super(RegressionModel, self).__init__()
        self.dropout = dropout
        self.low = low
        self.high = high
        self.fc_layer_1 = nn.Linear(300, 500)
        self.fc_layer_2 = nn.Linear(500, 500)
        self.fc_layer_3 = nn.Linear(500, 100)
        self.fc_layer_4 = nn.Linear(100, 9)
        self.output = nn.Linear(9, 1)
        self.dp = nn.Dropout(p=self.dropout)

        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(9)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.fc_layer_1(x)))
        x = self.dp(x)
        x = F.relu(self.bn2(self.fc_layer_2(x)))
        x = self.dp(x)
        x = F.relu(self.bn3(self.fc_layer_3(x)))
        x = self.dp(x)
        x = F.softmax(self.bn4(self.fc_layer_4(x)), dim=1)
        x = self.output(x)
        x = ((x-torch.min(x))/(torch.max(x)-torch.min(x)))*(self.max-self.min)+self.min
        return x
