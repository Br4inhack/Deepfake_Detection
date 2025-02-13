import torch.nn as nn
import torch

class AudioDeepfakeModel(nn.Module):
    def __init__(self):
        super(AudioDeepfakeModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 38 * 98, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.sigmoid(x)
