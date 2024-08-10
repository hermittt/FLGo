from flgo.utils import fmodule
import flgo
import numpy as np
import torch,copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from flgo.algorithm.fedbase import BasicServer, BasicClient
from flgo.my_algorithm.extra import extraServer, extraClient
from dataclasses import dataclass
import flgo.algorithm.fedavg as fedavg

from flgo.my_algorithm.my_utils import grad_False,grad_True,get_loc_data,save_bn_params,restore_bn_params

# based on official code https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/trainmodel/generator.py

class Generative_fedGEN(nn.Module):
  def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device) -> None:
      super().__init__()

      self.noise_dim = noise_dim
      self.num_classes = num_classes
      self.device = device
      self.fc1 = nn.Sequential(
          nn.Linear(noise_dim + num_classes, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.ReLU()
      )
      self.fc = nn.Linear(hidden_dim, feature_dim)

  def forward(self, labels):
      batch_size = labels.shape[0]
      eps = torch.rand((batch_size, self.noise_dim), device=self.device) # sampling from Gaussian
      y_input = F.one_hot(labels, self.num_classes)
      z = torch.cat((eps, y_input), dim=1)

      z = self.fc1(z)
      z = self.fc(z)
      return z

class Server(BasicServer):
  def initialize(self):
    BasicServer.initialize(self)
    self.init_algo_para({'server_epochs':500,'num_classes':10,'noise_dim':256,'generator_learning_rate':0.05,
              'localize_feature_extractor':False,'hidden_dim':256,'learning_rate_decay_gamma':0.99,'alpha':0.2})
    self.batch_size = self.option['batch_size']
    self.generative_model = Generative_fedGEN(self.noise_dim, self.num_classes, self.hidden_dim, self.model.fc.in_features, self.device).to(self.device)
    self.generative_optimizer = torch.optim.Adam(
        params=self.generative_model.parameters(),
        lr=self.generator_learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0, amsgrad=False)
    self.generative_learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=self.generative_optimizer, gamma=self.learning_rate_decay_gamma)
    self.loss = nn.CrossEntropyLoss()

    self.qualified_labels = []
    for client in self.clients:
      client.sample_per_class,client.train_samples = get_loc_data(client.id,self.option['task'])
      for yy in range(self.num_classes):
        self.qualified_labels.extend([yy for _ in range(int(client.sample_per_class[yy].item()))])
    for client in self.clients:
      client.qualified_labels = self.qualified_labels

  def pack(self, client_id, mtype=0, *args, **kwargs):
    return {
        "model": copy.deepcopy(self.model),
        "generative_model": copy.deepcopy(self.generative_model)
    }
  def iterate(self):
    self.selected_clients = self.sample()
    models = self.communicate(self.selected_clients)['model']
    self.train_generator(models,self.selected_clients)
    self.model = self.aggregate(models)
    return len(models) > 0

  def train_generator(self,uploaded_models,selected_clients):
    grad_True(self.generative_model)
    self.generative_model.train()
    uploaded_weights = self.get_uploaded_weights(selected_clients)
    for _ in range(self.server_epochs):
        labels = np.random.choice(self.qualified_labels, self.batch_size)
        labels = torch.LongTensor(labels).to(self.device)
        z = self.generative_model(labels)
        logits = 0
        for w, model in zip(uploaded_weights, uploaded_models):
            model.eval()
            if self.localize_feature_extractor:
                logits += model(z) * w
            else:
                logits += model.fc(z) * w

        self.generative_optimizer.zero_grad()
        loss = self.loss(logits, labels)
        loss.backward()
        self.generative_optimizer.step()

    self.generative_learning_rate_scheduler.step()

  def get_uploaded_weights(self,selected_clients):
    uploaded_weights = []
    tot_samples = 0
    for client_id in selected_clients:
      client = self.clients[client_id]
      tot_samples += client.train_samples
      uploaded_weights.append(client.train_samples)
    for i, w in enumerate(uploaded_weights):
      uploaded_weights[i] = w / tot_samples
    return uploaded_weights
class Client(extraClient):
  def unpack(self, received_pkg):
    self.generative_model = received_pkg['generative_model']
    return received_pkg['model']
  def initialize(self, *args, **kwargs):
    self.loss = nn.CrossEntropyLoss()
  def prepare_train(self,model):
    grad_False(self.generative_model)
    self.generative_model.eval()
  def local_training_with_extra_calculate(self, model, loss, outputs, batch_data):
    labels = np.random.choice(self.qualified_labels, self.batch_size)
    labels = torch.LongTensor(labels).to(self.device)
    with torch.no_grad():
      z = self.generative_model(labels)
    saved_bn_params = save_bn_params(model, self.teacher_model)
    z_loss = self.alpha * self.loss(model.fc(z.detach(), labels)
    restore_bn_params(model, saved_bn_params)    # 恢复model的原始BN层参数
    return loss + z_loss

class FedGen:
  Server=Server
  Client=Client
