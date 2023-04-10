import torchvision
from flgo.benchmark.toolkits.cv.classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
import flgo.benchmark
import os.path

builtin_class = torchvision.datasets.MNIST
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
path = os.path.join(flgo.benchmark.path,'RAW_DATA', 'MNIST')

TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path):
        super(TaskGenerator, self).__init__(__file__.split('/')[-2], rawdata_path, builtin_class, transform)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, builtin_class, transform)