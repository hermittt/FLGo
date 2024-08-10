import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import cv2 as cv

def save_bn_params(model, teacher):
    """
    保存 model 的 Batch Normalization 层参数。
    Returns:
        dict: 保存的 Batch Normalization 层参数。
    """
    saved_bn_params = {}
    for (name, module), (_, teacher_module) in zip(model.named_modules(), teacher.named_modules()):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # 保存当前 model 的 Batch Normalization 层参数
            saved_bn_params[name] = {
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone(),
            }
    return saved_bn_params

def restore_bn_params(model, saved_bn_params):
    """
    恢复 model 中保存的 Batch Normalization 层参数。

    Args:
        model (nn.Module): 目标模型。
        saved_bn_params (dict): 之前保存的 Batch Normalization 层参数。
    """
    for name, module in model.named_modules():
        if name in saved_bn_params:
            # 恢复之前保存的参数
            module.running_mean = saved_bn_params[name]['running_mean']
            module.running_var = saved_bn_params[name]['running_var']

def get_Normalize_mean_std(transform):
  for t in transform.transforms:
    if isinstance(t, transforms.Normalize):
        return t.std,t.mean
def get_transform(dataset):
  if hasattr(dataset, 'transform'):
    return dataset.transform
  else: #针对嵌套很多层的情况，递归调用来寻找dataset中的transform
    return get_transform(dataset.dataset)

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
                torch.mean(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch),dim=1))
    else:
      loss = T * T * \
                torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch),dim=1)
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

def merge(images, size): #https://github.com/znxlwm/pytorch-generative-model-collections/blob/my/utils.py
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')
def img_frame(samples,image_frame_dim,transform):
  samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
  if transform!=None:
    std,mean=get_Normalize_mean_std(transform)
    samples = samples*std+mean #反归一化
  samples = np.squeeze(merge(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim] ))
  if samples.ndim == 2:
    samples = np.expand_dims(samples, axis=2)
  return samples
def show_img(samples,image_frame_dim,path,x=None,transform=None):
  img_float = img_frame(samples,image_frame_dim,transform)
  if x!=None:
    x_float = img_frame(x,image_frame_dim,transform)
    imsize = int(x_float.shape[0]/image_frame_dim)
    zero_list = np.ones((x_float.shape[0], 2, x_float.shape[2])).tolist()
    img_float = np.concatenate((x_float[:,:imsize,:],zero_list, img_float), axis=1)
  image = np.clip((img_float * 255),0, 255).astype(np.uint8)
  # 将 RGB 图像数据转换为 BGR 顺序, 因为OpenCV 中，默认的颜色通道顺序是 BGR（蓝绿红），而不是常见的 RGB（红绿蓝）
  image2 = cv.cvtColor(image, cv.COLOR_RGB2BGR)
  cv.imwrite(path, image2)
  return image

def img_change(x,transform): #[-1, 1]将范围内的值转换为[0, 1]范围内的值的过程
  std,mean=get_Normalize_mean_std(transform)
  if type(std)==tuple:
    std,mean=torch.tensor(std).to(x.device).view(1,-1,1,1),torch.tensor(mean).to(x.device).view(1,-1,1,1)
  x = x*0.5+0.5
  return (x-mean)/std.clamp(min=1e-5)
