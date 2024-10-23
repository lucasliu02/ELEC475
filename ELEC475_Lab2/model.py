import torch.nn.functional as F
import torch.nn as nn
import torch

class SnoutNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64,3, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.input_shape = (3, 227, 227)
        self.mp = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.mp(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
