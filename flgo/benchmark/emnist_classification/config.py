import torchvision
import os

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.1307,), (0.3081,))]
)
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'RAW_DATA', 'EMNIST')
train_data = torchvision.datasets.EMNIST(root=path, split='letters',train=True, download=True, transform=transform)
test_data = torchvision.datasets.EMNIST(root=path, split='letters', train=False, download=True, transform=transform)
