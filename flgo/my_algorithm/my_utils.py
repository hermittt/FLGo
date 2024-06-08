import json
import torch
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