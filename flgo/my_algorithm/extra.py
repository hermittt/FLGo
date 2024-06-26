from flgo.utils import fmodule
import flgo
import torch,copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from flgo.algorithm.fedbase import BasicServer, BasicClient

from flgo.my_algorithm.my_utils import grad_False,grad_True


class extraServer(BasicServer):
  def iterate(self):
    self.selected_clients = self.sample()
    recv = self.communicate(self.selected_clients)
    models = recv['model']
    self.extra_received(models,recv)
    self.model = self.aggregate(models)
    self.extra_after()
    return len(models) > 0
  def extra_received(self,models,recv):
    pass #处理收到的额外信息
  def extra_after(self):
    pass
class extraClient(BasicClient):
  def unpack(self, received_pkg):
    self.extra_received(received_pkg)
    return received_pkg['model']
  def extra_received(self, received_pkg):
    pass #其他都直接写到self.里
  def train(self, model):
    grad_True(model)
    model.train()
    self.prepare_train(model)
    optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
    self.trainning_output={} # 定义一个空字典,在任意位置添加当前需要输出的内容(如self.trainning_output['loss'] = loss.item())
    for iter in range(self.num_steps):
      batch_data = self.get_batch_data()
      model.zero_grad()
      computed_loss = self.calculator.compute_loss(model, batch_data)
      loss, outputs= computed_loss['loss'],computed_loss['outputs']
      loss = self.local_training_with_extra_calculate(model,loss,outputs,batch_data)
      loss.backward()
      if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
      optimizer.step()
      if (iter+1) % 50 == 0:
        # 按固定格式将每一个元素连接成一个字符串，并输出
        output = ", ".join([f"{key}：{value:.2f}" if isinstance(value, float) 
                else f"{key}：{value}" for key, value in self.trainning_output.items()])
        print(output)
    self.after_train(model)
    return
  def prepare_train(self,model):
    pass #设置一些辅助模型的参数冻结等
  def after_train(self,model):
    pass #处理一些输出等
  def local_training_with_extra_calculate(self, model, loss, outputs, batch_data):
    return loss

class extra:
  Server=extraServer
  Client=extraClient
