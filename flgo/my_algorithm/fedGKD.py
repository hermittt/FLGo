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
from flgo.my_algorithm.my_utils import grad_False,grad_True,KL_Loss_equivalent


class GKDServer(extraServer): #FedGKD，FedKF通用，传输额外的缓存模型
  def initialize(self):
    self.init_algo_para({'local':'ACA','teacher':1,'buffer_len':0,'T':'1','esb_w':5.0,'distill_w':'0.001*self.round'})
    #local(本地模型)=ACA(当前),OCA(缓存),teacher=0(和local一样),1(两个模型一起，双倍通信),buffer_len:<=0按类别，>0最近k个
    if self.local=='ACA' and self.teacher==0: #无缓存
      pass
    if self.buffer_len > 0:
      self.buffer = []
    else:
      self.buffer = [[i, None] for i in range(len(self.clients))] #初始化空的客户端模型列表
  def extra_received(self,models,recv):
    if self.local=='ACA' and self.teacher==0: #无缓存
      pass
    if self.buffer_len <= 0:
      for client_id,client_model in zip(self.selected_clients,models):
        grad_False(client_model)
        client_model.eval()
        self.buffer[client_id][1]=client_model
  def extra_after(self):
    if self.local=='ACA' and self.teacher==0: #无缓存
      pass
    if self.buffer_len > 0:
      if len(self.buffer) >= self.buffer_len:
        self.buffer.pop(0)
      self.buffer.append([0,copy.deepcopy(self.model)]) #0没有用，纯接口占位
  def pack(self, client_id, mtype=0, *args, **kwargs):
    if self.local=='ACA' and self.teacher==0: #无缓存
      return {
          "model": copy.deepcopy(self.model),
          "round": self.current_round,
      }
    ensemble_models = []
    for client_idx, local_model in self.buffer:
      if local_model == None:
        continue
      else:
        ensemble_models.append(local_model)
    ensemble_model = self.aggregate(ensemble_models)
    if self.teacher==1: #两个都传 
      return {
          "model": copy.deepcopy(self.model),
          "ensemble_model": copy.deepcopy(ensemble_model),
          "round": self.current_round,
      }
    else:
      return {
          "model": copy.deepcopy(ensemble_model),
          "round": self.current_round,
      }
class GKDClient(extraClient):
  def initialize(self, *args, **kwargs):
    self.ensemble_model = None
    self.loss = nn.CrossEntropyLoss()
    self.KL_loss = KL_Loss_equivalent()
  def extra_received(self, received_pkg):
    if self.teacher==1: #两个都传 
      self.ensemble_model = received_pkg['ensemble_model']
    self.round = received_pkg['round']
  def prepare_train(self,model):
    self.teacher_model = copy.deepcopy(model)
    grad_False(self.teacher_model)
    self.teacher_model.eval()
    if self.teacher==1: #两个都传 
      grad_False(self.ensemble_model)
      self.ensemble_model.eval()
  def local_training_with_extra_calculate(self, model, loss, outputs, batch_data):
    x, y = batch_data
    x = x.to(self.device)
    C_teacher = self.teacher_model(x).detach()
    if self.teacher==0: #只有一个teacher
      distill_loss = self.KL_loss(C_teacher,outputs,T=eval(self.T))
    if self.teacher==1: #两个都传 
      C_ensemble = self.ensemble_model(x).detach()
      distill_loss = self.KL_loss(C_teacher,outputs,T=eval(self.T))+self.esb_w*self.KL_loss(C_ensemble,outputs,T=eval(self.T))
    return loss + distill_loss * eval(self.distill_w)
class FedGKD:
  Server=GKDServer
  Client=GKDClient
