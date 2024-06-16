import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class KL_Loss_equivalent(nn.Module):
  def __init__(self,softmax_fn=True):
    super(KL_Loss_equivalent, self).__init__()
    self.softmax_fn=softmax_fn
  def forward(self, output_batch, teacher_outputs, T=8, reduce=True):
    if self.softmax_fn:
      output_batch = F.log_softmax(output_batch / T, dim=1)
      teacher_outputs = F.softmax(teacher_outputs / T, dim=1) + 10 ** (-7)
    if reduce==True:
      loss = T * T * \
                  torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
    else:
      loss = T * T * \
                torch.mean(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch),dim=1))
    return loss
  
def grad_False(model, select_frozen_layers=None):
    if select_frozen_layers==None:
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        i = 0
        for name, param in model.named_parameters():
            if select_frozen_layers in model.named_parameter_layers[i]:
                param.requires_grad = False
            i += 1

def grad_True(model):
    for name, param in model.named_parameters():
        param.requires_grad = True

def get_loc_data(client_id,task):
  with open(task+'/loc_data.json', 'r') as file:
    json_data = json.load(file)
  class_info = json_data[str(client_id)]
  date_num=class_info["date_num"]
  c = torch.tensor(class_info["calss_num"])
  print("client",client_id,"总样本数：",date_num," 各类：", c.int().numpy())
  return c,date_num