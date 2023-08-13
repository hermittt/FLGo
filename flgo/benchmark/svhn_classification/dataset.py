import os
import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RAW_DATA', 'SVHN')
train_data = torchvision.datasets.SVHN(root=path,transform=transform, download=True, split='train')
test_data = torchvision.datasets.SVHN(root=path, transform=transform, download=True, split='test')