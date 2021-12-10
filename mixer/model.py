import torch
import torch.nn as nn
import torch.nn.functional as F

class Assembler(nn.Module):
    def __init__(self, hidden_dim):
        super(Assembler, self).__init__()
        self.transformation_params = 3 * 5
        self.fc1 = nn.Linear(hidden_dim*5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.scale_head = nn.Linear(256, self.transformation_params)
        self.trans_head = nn.Linear(256, self.transformation_params)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        scale = self.scale_head(x).reshape(-1, 5, 3)
        trans = self.trans_head(x).reshape(-1, 5, 3)
        transformation = torch.cat((scale, trans), -1)
        return transformation
