from torch import nn
import torch.nn.functional as F
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)  # 添加 BatchNorm
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)  # 添加 BatchNorm
        self.fc1 = nn.Linear(3136, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)  # 添加 BatchNorm
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc(x)
        return x

    def get_embedding(self, x):
        x = x.view((x.shape[0],28,28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)  # BatchNorm应用在激活函数之前
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)  # BatchNorm应用在激活函数之前
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.bn_fc1(self.fc1(x)))  # BatchNorm应用在激活函数之前
        return x
        
def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)