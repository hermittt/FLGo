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
from collections import defaultdict
#https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/clients/clientdistill.py
class Server(extraServer):
  def initialize(self):
    BasicServer.initialize(self)
    self.init_algo_para({'lamda':1.0,'num_classes':10})
    self.global_logits = [None for _ in range(self.num_classes)]
  def extra_received(self,models,recv):
    self.global_logits = self._logit_aggregation(recv["logits"])
  def pack(self, client_id, mtype=0, *args, **kwargs):
    return {
        "model": copy.deepcopy(self.model),
        "global_logits": copy.deepcopy(self.global_logits)
    }
  # https://github.com/yuetan031/fedlogit/blob/main/lib/utils.py#L221
  def _logit_aggregation(self,local_logits_list):
    agg_logits_label = defaultdict(list)
    for local_logits in local_logits_list:
        for label in local_logits.keys():
            agg_logits_label[label].append(local_logits[label])

    for [label, logit_list] in agg_logits_label.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            agg_logits_label[label] = logit / len(logit_list)
        else:
            agg_logits_label[label] = logit_list[0].data

    return agg_logits_label
class Client(extraClient):
  def initialize(self, *args, **kwargs):
    self.global_logits = None
    self.loss = nn.CrossEntropyLoss()
  def unpack(self, received_pkg):
    self.global_logits = received_pkg['global_logits']
    self.logits = defaultdict(list)
    return received_pkg['model']
  def local_training_with_extra_calculate(self, model, loss, outputs, batch_data):
    x, y = batch_data
    if self.global_logits != None:
      logit_new = copy.deepcopy(outputs.detach())
      for i, yy in enumerate(y):
        y_c = yy.item()
        if type(self.global_logits[y_c]) != type([]) and self.global_logits[y_c] != None:
          logit_new[i, :] = self.global_logits[y_c].data
      loss += self.loss(outputs, logit_new.softmax(dim=1)) * self.lamda
    for i, yy in enumerate(y):
      y_c = yy.item()
      self.logits[y_c].append(outputs[i, :].detach().data)
    return loss
  def pack(self, model, *args, **kwargs):
    return {
        "model": model,
        "logits": copy.deepcopy(self._agg_func(self.logits))
    }
  def _agg_func(self,logits):
    for [label, logit_list] in logits.items():
      if len(logit_list) > 1:
          logit = 0 * logit_list[0].data
          for i in logit_list:
              logit += i.data
          logits[label] = logit / len(logit_list)
      else:
          logits[label] = logit_list[0]
    return logits
class FedDistill:
  Server=Server
  Client=Client