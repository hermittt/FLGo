from torch import nn
import torch.nn.functional as F
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(num_groups=1, num_channels=32)  # 将 BatchNorm2d 替换为 GroupNorm
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.gn2 = nn.GroupNorm(num_groups=1, num_channels=64)  # 将 BatchNorm2d 替换为 GroupNorm
        self.fc1 = nn.Linear(3136, 512)
        self.fc = nn.Linear(512, 26)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc(x)
        return x

    def get_embedding(self, x):
        x = x.view((x.shape[0], 28, 28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.gn1(self.conv1(x))), 2)  # 将 BatchNorm 替换为 GroupNorm
        x = F.max_pool2d(F.relu(self.gn2(self.conv2(x))), 2)  # 将 BatchNorm 替换为 GroupNorm
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        return x
        
def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)
