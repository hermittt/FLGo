from flgo.utils import fmodule
import flgo
import torch,copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from flgo.algorithm.fedbase import BasicServer, BasicClient
from dataclasses import dataclass
import flgo.algorithm.fedavg as fedavg

from flgo.my_utils import grad_False,grad_True,get_loc_data


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
  def initialize(self):
    BasicServer.initialize(self)
    args = self.option['args']
    # self.load_model()
    self.Budget = []
    self.batch_size = option['batch_size']
    self.generative_model = Generative_fedGEN(args.noise_dim, args.num_classes, args.hidden_dim, self.model.fc.in_features, self.device).to(self.device)
    self.generative_optimizer = torch.optim.Adam(
        params=self.generative_model.parameters(),
        lr=args.generator_learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0, amsgrad=False)
    self.generative_learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=self.generative_optimizer, gamma=args.learning_rate_decay_gamma)
    self.loss = nn.CrossEntropyLoss()

    self.qualified_labels = []
    for client in self.clients:
      client.sample_per_class,client.train_samples = get_loc_data(client.id,self.option['task'])
      for yy in range(args.num_classes):
        self.qualified_labels.extend([yy for _ in range(int(client.sample_per_class[yy].item()))])
    for client in self.clients:
      client.qualified_labels = self.qualified_labels

    self.server_epochs = args.server_epochs
    self.localize_feature_extractor = args.localize_feature_extractor
    if self.localize_feature_extractor:
      self.global_model = copy.deepcopy(args.model.fc)

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
class Client(BasicClient):
  def unpack(self, received_pkg):
    return received_pkg['model'],received_pkg['generative_model']
  def reply(self, svr_pkg):
    model,generative_model = self.unpack(svr_pkg) #svr_pkg (dict): the package received from the server
    self.train(model,generative_model)
    cpkg = self.pack(model)
    return cpkg #client_pkg (dict): the package to be send to the server
  def initialize(self, *args, **kwargs):
    self.args = self.option['args']
    self.localize_feature_extractor = self.args.localize_feature_extractor
    self.loss = nn.CrossEntropyLoss()

  def train(self, model, generative_model):
    grad_True(model)
    model.train()
    grad_False(generative_model)
    generative_model.eval()
    optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
    for iter in range(self.num_steps):
      batch_data = self.get_batch_data()
      model.zero_grad()
      loss = self.calculator.compute_loss(model, batch_data)['loss']
      x, y = batch_data
      x.to(device=self.device)
      labels = np.random.choice(self.qualified_labels, self.batch_size)
      labels = torch.LongTensor(labels).to(self.device)
      with torch.no_grad():
        z = generative_model(labels)
      loss += self.args.alpha * self.loss(model.fc(z), labels)
      loss.backward()
      if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
      optimizer.step()
    return
class FedGen:
  Server=Server
  Client=Client