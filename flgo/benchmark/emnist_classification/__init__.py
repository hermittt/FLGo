import flgo.benchmark.toolkits.visualization
from flgo.benchmark.emnist_classification.model import cnn_no_bn
import flgo.benchmark.toolkits.partition

default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
default_model = cnn_no_bn
visualize = flgo.benchmark.toolkits.visualization.visualize_by_class
