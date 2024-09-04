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
  def init_extra(self):#额外参数和其他初始化
    # 创建一个字典来存储所有参数
    algo_params = {  #local(本地模型)=ACA(当前),OCA(缓存),teacher=0(和local一样),1(两个模型一起，双倍通信),buffer_len:<=0按类别，>0最近k个
        'local': 'ACA',
        'teacher': 0,
        'show_fn': 1,
        'buffer_len': 0,
        'T': '10',
        'esb_w': 1.0,
        'distill_w1': '0.1*self.round',
        'min_round': 5,
    }
    algo_params = self.set_params(algo_params) #设置自定义参数
    # 初始化算法参数
    self.init_algo_para(algo_params)
  def set_params(self,algo_params):
    return algo_params
  def initialize(self):
    self.init_extra()
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
  def unpack(self, received_pkg):
    self.teacher_model = received_pkg['model']
    self.round = received_pkg['round']
    if self.teacher==0: #只有一个teacher
      return copy.deepcopy(received_pkg['model'])
    else: #两个都传 
      self.ensemble_model = received_pkg['ensemble_model']
      if self.local=='ACA':
        return copy.deepcopy(received_pkg['model'])
      else:
        return copy.deepcopy(received_pkg['ensemble_model'])
  def prepare_train(self,model):
    grad_False(self.teacher_model)
    self.teacher_model.eval()
    if self.teacher==1: #两个都传 
      grad_False(self.ensemble_model)
      self.ensemble_model.eval()

  def prepare_train(self,model):
    grad_False(self.teacher_model)
    self.teacher_model.eval()
    if self.teacher==1: #两个都传
      grad_False(self.ensemble_model)
      self.ensemble_model.eval()
    self.step0_flag = 1

  def local_training_with_extra_calculate(self, model, loss, outputs, batch_data):
    x, y = batch_data
    x,y = x.to(self.device),y.to(self.device)
    '''
    if self.step0_flag == 1:
      self.step0_flag = 0
      y_pre = outputs.max(1)[1]
      matches = y_pre == y # 比较两个tensor是否相等
      self.init_accuracy = matches.sum().item() / len(y)
      print(self.init_accuracy)
    '''
    if self.round>self.min_round:
      distill_loss = self.cal_L_kl(x,outputs)[0]
      return loss + distill_loss * eval(self.distill_w1)#*max(0.001,self.init_accuracy)
    else:
      return loss
  
  def G_acc(self,output,y_G):
    y_G_pre = output.max(1)[1]
    matches = y_G_pre == y_G # 比较两个tensor是否相等
    return matches.sum().item() / len(y_G),matches
    
  def cal_L_kl(self,x,C_student,reduce=True):
    with torch.no_grad():
      C_teacher = self.teacher_model(x).detach()
    if self.teacher==0: #一个teacher
      distill_loss = self.KL_loss(C_teacher.detach(),C_student,T=eval(self.T),reduce=reduce)
      return distill_loss,F.softmax(C_teacher)
    if self.teacher==1: #两个teacher 
      with torch.no_grad():
        C_ensemble = self.ensemble_model(x).detach()
      distill_loss = (self.KL_loss(C_teacher.detach(),C_student,T=eval(self.T),reduce=reduce)\
              +self.esb_w*self.KL_loss(C_ensemble.detach(),C_student,T=eval(self.T),reduce=reduce))/(1+self.esb_w)
      return distill_loss,(F.softmax(C_teacher)+F.softmax(C_ensemble))/2
class FedGKD:
  Server=GKDServer
  Client=GKDClient

class GKDServer_d1w001(GKDServer):
  def set_params(self,algo_params):
    algo_params['distill_w1'] = '0.01*self.round'
    return algo_params
class FedGKD_d1w001:
  Server=GKDServer_d1w001
  Client=GKDClient

class GKDServer_d1w0001(GKDServer):
  def set_params(self,algo_params):
    algo_params['distill_w1'] = '0.001*self.round'
    return algo_params
class FedGKD_d1w0001:
  Server=GKDServer_d1w0001
  Client=GKDClient

class GKDServer_d1w00001(GKDServer):
  def set_params(self,algo_params):
    algo_params['distill_w1'] = '0.0001*self.round'
    return algo_params
class FedGKD_d1w00001:
  Server=GKDServer_d1w00001
  Client=GKDClient

class GKDServer_d1w000001(GKDServer):
  def set_params(self,algo_params):
    algo_params['distill_w1'] = '0.00001*self.round'
    return algo_params
class FedGKD_d1w000001:
  Server=GKDServer_d1w000001
  Client=GKDClient

class GKDServer_d1w0(GKDServer):
  def set_params(self,algo_params):
    algo_params['distill_w1'] = '0.0'
    return algo_params
class FedGKD_d1w0:
  Server=GKDServer_d1w0
  Client=GKDClient
