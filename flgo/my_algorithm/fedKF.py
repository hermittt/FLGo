import os
import sys
import wandb
from flgo.utils import fmodule
import flgo
import numpy as np
import torch,copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from torch.utils.data import DataLoader
from flgo.my_algorithm.fedGKD import GKDServer, GKDClient
from flgo.my_algorithm.my_utils import grad_False,grad_True,KL_Loss_equivalent,get_loc_data,get_Normalize_mean_std,\
get_transform,merge,show_img,img_change
from flgo.my_algorithm.my_models import Encoder,Decoder,conv_layer
import threading

def img_change(x,transform): #[-1, 1]将范围内的值转换为[0, 1]范围内的值的过程
  std,mean=get_Normalize_mean_std(transform)
  if type(std)==tuple:
    std,mean=torch.tensor(std).to(x.device).view(1,-1,1,1),torch.tensor(mean).to(x.device).view(1,-1,1,1)
  std,mean=torch.tensor(std).to(x.device),torch.tensor(mean).to(x.device)
  x = x*0.5+0.5
  return (x-mean)/std.clamp(min=1e-5)

def generate_labels(class_num, batch_size, rng_local=np.random.RandomState(0), mod="regu"): #随机种子rng_local
  weight = class_num * [float(1.0 / class_num)]
  if mod == "regu":
    mean_Y = torch.range(0, class_num - 1, dtype=torch.int64).repeat(batch_size // class_num)
    y_disc = torch.from_numpy(rng_local.randint(0, class_num - 1, size=[batch_size % class_num]))
    y_disc_ = torch.cat([mean_Y, y_disc], 0)
  else:
    y_disc_ = torch.from_numpy(rng_local.randint(0, class_num - 1, size=[batch_size]))
  return y_disc_

def out_feature(md,x):
  features = md.get_embedding(x)
  output = md.fc(features)
  return output,features

class SCELoss(torch.nn.Module):
  # https://github.com/FangXiuwen/Robust_FL/blob/master/loss.py
  def __init__(self, alpha=1.0, beta=0.1, num_classes=10):
    super(SCELoss, self).__init__()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.alpha = alpha
    self.beta = beta
    self.num_classes = num_classes
    self.cross_entropy = torch.nn.CrossEntropyLoss()

  def forward(self, pred, labels, mod='sce'):
    # CCE
    ce = self.cross_entropy(pred, labels)
    if mod=='sce':
      # RCE
      pred = F.softmax(pred, dim=1)
      pred = torch.clamp(pred, min=1e-7, max=1.0)
      label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
      label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
      rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
      # Loss
      loss = self.alpha * ce + self.beta * rce.mean()
      return loss


class Generative(nn.Module):
  # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
  # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
  def __init__(self, input_dim, image_channels, class_num, latent_dim, img_size, device, transform=None, VQ=None) -> None:
    super().__init__()
    self.input_dim = input_dim
    self.image_channels = image_channels
    self.img_size = img_size
    self.class_num = class_num
    self.transform = transform
    self.device = device
    self.latent_dim = latent_dim
    self.vqgan = VQ
    self.fc = nn.Sequential(
        nn.Linear(self.input_dim, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, latent_dim * (self.img_size // 4) * (self.img_size // 4)),
        nn.BatchNorm1d(latent_dim * (self.img_size // 4) * (self.img_size // 4)),
        nn.ReLU(),
    )
    self.conv = nn.Sequential(
        nn.ReflectionPad2d(1),#填充和周围相似的
        nn.Conv2d(self.latent_dim, 128, 3, 1),
        conv_layer(128,128),
        conv_layer(128,128),
        conv_layer(128,128),
        nn.Conv2d(128, self.latent_dim, 1, 1, 0),
        nn.Tanh(),
    )
    @dataclass
    class Namespace:
        latent_dim: int = self.latent_dim
        image_channels: int = self.image_channels
        out_kernel: tuple =(3,1,1)
        device: str = self.device
    args = Namespace()
    self.decoder = Decoder(args)
    self.embedding = nn.Embedding(self.class_num, self.input_dim)

  def forward(self, input, y):
    label_embedding = self.embedding(y)
    h = torch.mul(input.float(), label_embedding.float())
    z = self.fc(h)
    z = z.view(-1, self.latent_dim, (self.img_size // 4), (self.img_size // 4))
    #x = self.conv(x)
    if self.vqgan!=None:
      z, _, _ = self.vqgan.codebook(z)
      post_quant_conv_mapping = self.vqgan.post_quant_conv(z)
      x = self.vqgan.decoder(post_quant_conv_mapping)
    else:
      x = self.decoder(z)
    mean,var = x.mean(),x.var()
    if self.transform!=None:
      return img_change(x,self.transform),mean,var
    else:
      return x,mean,var

class KFServer(GKDServer): #FedGKD，FedKF通用，传输额外的缓存模型
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
        'distill_w2': '0.1*self.round',
        'min_round': 5,
        'generator_learning_rate': 5e-4,
        'act_w': 1e-4,
        'noise_dim': 100,
        'VQ': 0,
        'img_size': self.option['imsz'],
        'img_channels': self.option['imch'],
        'latent_dim': self.option['latent_dim'],
        'num_classes': self.option['num_classes'],
        'transform': get_transform(self.test_data),
        'c_loss_type': 'SCE', #'SCE'或者单纯的'CE',
        'init_acc':1,
        'self.save_md':0
    }
    algo_params = self.set_params(algo_params) #设置自定义参数
    # 初始化算法参数
    self.init_algo_para(algo_params)
  def set_params(self,algo_params):
    return algo_params
class KFClient(GKDClient):
  def initialize(self, *args, **kwargs):
    self.extra_init()
    self.ensemble_model = None
    self.loss = SCELoss(num_classes=self.num_classes) #nn.CrossEntropyLoss()
    self.KL_loss = KL_Loss_equivalent()
    self.generative_optimizer = torch.optim.Adam(
                  params=self.G.parameters(),
                  lr=self.generator_learning_rate, betas=(0.9, 0.999),
                  eps=1e-08, weight_decay=0, amsgrad=False)
    self.local_step = 0
    #输出图像用
    self.rng_local = np.random.RandomState(0) #专用的随机对象
    self.rslt_path = self.get_rslt_path()
    if not os.path.exists(self.rslt_path):
      os.makedirs(self.rslt_path)
    sample_y_=generate_labels(self.num_classes, self.num_classes)
    self.sample_y_=sample_y_.unsqueeze(0).repeat(self.num_classes,1,1).flatten().to(self.device)
  def extra_init(self):
    if self.VQ==1:
      vqgan = get_model(self.option['vqgan_args'],self.option['task'],self.id)
    else:
      vqgan = None
    self.G = Generative(self.noise_dim, self.img_channels, self.num_classes, self.latent_dim, \
                  self.img_size, self.device ,transform=self.transform, VQ=vqgan).to(self.device)
    sample_z_=torch.tensor(np.random.normal(0, 1, (self.num_classes, self.noise_dim)))
    self.sample_z_=sample_z_.unsqueeze(1).repeat(1,self.num_classes,1).reshape(-1,self.noise_dim).to(self.device)
  def get_rslt_path(self):
    task = self.option['task'].replace('/','').replace('.','').replace('task','')
    self.sample_per_class,_ = get_loc_data(self.id,self.option['task'])
    c = self.sample_per_class.topk(k=10).indices
    name='client '+str(self.id)+str(c.tolist())
    return 'result_G/' + task + '/' + name

  def local_training_with_extra_calculate(self, model, loss, outputs, batch_data):
    grad_False(self.G)
    self.G.eval()
    x, y = batch_data
    x,y = x.to(self.device),y.to(self.device)
    if self.step0_flag == 1:
      self.step0_flag = 0
      y_pre = outputs.max(1)[1]
      matches = y_pre == y # 比较两个tensor是否相等
      self.init_accuracy = matches.sum().item() / len(y)
      print(self.init_accuracy)
    if self.round>self.min_round :
      y_G = generate_labels(self.num_classes, y.shape[0], rng_local=self.rng_local).to(self.device)
      with torch.no_grad():
        G = self.generate_and_train_generator(x,y,y_G,train=False)
      if eval(self.distill_w1)>0:
        distill_loss = self.cal_L_kl(x,outputs)[0]
      G_distill_loss,outputs_G = self.cal_L_kl(G.detach(),model(G.detach()),reduce=False)
      G_accuracy = self.G_acc(outputs_G,y_G)[1]
      w = max(0.001, self.init_accuracy) if self.init_acc else 1
      if eval(self.distill_w1)>0:
        return loss + (distill_loss*eval(self.distill_w1)+(G_distill_loss*G_accuracy).mean()*eval(self.distill_w2))*w
      else:
        return loss + ((G_distill_loss*G_accuracy).mean()*eval(self.distill_w2))*w
    else:
      return loss

  def after_iter(self,model,batch_data):
    x, y = batch_data
    x,y = x.to(self.device),y.to(self.device)
    y_G = generate_labels(self.num_classes, y.shape[0], rng_local=self.rng_local).to(self.device)
    if y.shape[0]>1 and self.round>self.min_round:
      self.generate_and_train_generator(x,y,y_G)
      
  def generate_and_train_generator(self,x,y,y_G,train=True): #生成图像，并执行一步生成模型的训练step
    z_G = torch.tensor(np.random.normal(0, 1, (y.shape[0], self.noise_dim))).to(self.device).float()
    if train==False:
      return self.G(z_G,y_G)[0]
    grad_True(self.G)
    self.G.train()
    if self.VQ==1:
      grad_False(self.G.vqgan)
      self.G.vqgan.eval()
    #self.bn_loss.bn_clear() #bn_loss 3
    self.generative_optimizer.zero_grad()
    G = self.G(z_G,y_G)[0]
    mean,var = G.mean(),G.var()
    output,features = out_feature(self.teacher_model,G)
    
    if self.c_loss_type=='CE':
      c_loss = F.cross_entropy(output, y_G)
      if self.teacher==1:
        c_loss = (c_loss + self.esb_w*F.cross_entropy(output, y_G))/(1+self.esb_w)
    else:
      c_loss = self.loss(output, y_G) #+ self.loss(output3, y)
      if self.teacher==1:
        c_loss = (c_loss + self.esb_w*self.loss(output, y_G))/(1+self.esb_w)
    
    self.trainning_output['c_loss'] = c_loss.item()
    loss_activation = -features.abs().mean()
    #bn_ls = self.bn_loss.bn_ls()#bn_loss 4
    loss = c_loss + self.act_w*loss_activation #+ bn_ls
    loss.backward()
    if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=self.G.parameters(), max_norm=self.clip_grad)
    self.generative_optimizer.step()
    return
  def after_train(self,model):
    self.local_step+=1
    if self.show_fn==1 and self.round>self.min_round:
      grad_False(self.G)
      self.G.eval()
      G = self.G(self.sample_z_,self.sample_y_)[0]
      self.show([G])
      if self.save_md==1:
        self.save_G()
  def save_G(self):
      # 保存 self.G 模型到一个文件，例如 'model_G.pth'
      torch.save(self.G.state_dict(), "model_G.pth")
      # 创建 Artifact 并将模型文件加入
      artifact = wandb.Artifact('model_G %2d %2d' % (self.round,self.id), type='model')
      artifact.add_file('model_G.pth')
      # 将 Artifact 上传到 wandb
      wandb.log_artifact(artifact)
            
  def show(self,imgs,x=None):
    name='client '+str(self.id)
    imgs_out=[]
    for i, img in enumerate(imgs):
      path=self.rslt_path + '/%2d[%d]' % (self.round,i) + '.png'
      image=show_img(img,self.num_classes,path,x=x,transform=self.transform)
      imgs_out.append(wandb.Image(image, caption=[self.round,i]))
    t = threading.Thread(target=wandb.log, args=({name: imgs_out},))
    t.start()
class FedKF:
  Server=KFServer
  Client=KFClient

class KFServer_CE(KFServer): #mnist-a0.1
  def set_params(self,algo_params):
    algo_params['c_loss_type'] = 'CE'
    return algo_params
class FedKF_CE:
  Server=KFServer_CE
  Client=KFClient

class KFServer_CE_d1w0001(KFServer): #mnist-a0.1
  def set_params(self,algo_params):
    algo_params['c_loss_type'] = 'CE'
    algo_params['distill_w1'] = '0.001*self.round'
    return algo_params
class FedKF_CE_d1w0001:
  Server=KFServer_CE_d1w0001
  Client=KFClient

class KFServer_CE_d1w0(KFServer): #mnist-a0.1
  def set_params(self,algo_params):
    algo_params['c_loss_type'] = 'CE'
    algo_params['distill_w1'] = '0'
    return algo_params
class FedKF_CE_d1w0:
  Server=KFServer_CE_d1w0
  Client=KFClient
  
class KFServer_CE0(KFServer): #mnist-a0.1
  def set_params(self,algo_params):
    algo_params['c_loss_type'] = 'CE'
    algo_params['init_acc'] = 0
    return algo_params
class FedKF_CE0:
  Server=KFServer_CE0
  Client=KFClient

class KFServer_CE01(KFServer): #mnist-a0.1
  def set_params(self,algo_params):
    algo_params['c_loss_type'] = 'CE'
    algo_params['init_acc'] = 0
    algo_params['save_md'] = 1
    return algo_params
class FedKF_CE01:
  Server=KFServer_CE01
  Client=KFClient

class KFServer_CE0_d1w0001(KFServer): #mnist-a0.1
  def set_params(self,algo_params):
    algo_params['c_loss_type'] = 'CE'
    algo_params['distill_w1'] = '0.001*self.round'
    algo_params['init_acc'] = 0
    return algo_params
class FedKF_CE0_d1w0001:
  Server=KFServer_CE0_d1w0001
  Client=KFClient

class KFServer_CE0_d1w0(KFServer): #mnist-a0.1
  def set_params(self,algo_params):
    algo_params['c_loss_type'] = 'CE'
    algo_params['distill_w1'] = '0'
    algo_params['init_acc'] = 0
    return algo_params
class FedKF_CE0_d1w0:
  Server=KFServer_CE0_d1w0
  Client=KFClient
  

class KFServer_d1w0(KFServer): #mnist-a0.1
  def set_params(self,algo_params):
    algo_params['distill_w1'] = '0.0'
    return algo_params
class FedKF_d1w0:
  Server=KFServer_d1w0
  Client=KFClient


class KFServer_d1w001(KFServer): #mnist-a0.1
  def set_params(self,algo_params):
    algo_params['distill_w1'] = '0.01*self.round'
    return algo_params
class FedKF_d1w001:
  Server=KFServer_d1w001
  Client=KFClient

class KFServer_d1w0001(KFServer): #mnist-a0.01
  def set_params(self,algo_params):
    algo_params['distill_w1'] = '0.001*self.round'
    return algo_params
class FedKF_d1w0001:
  Server=KFServer_d1w0001
  Client=KFClient

class KFServer_d1w0001_d2w001(KFServer): #mnist-a0.01
  def set_params(self,algo_params):
    algo_params['distill_w1'] = '0.001*self.round'
    algo_params['distill_w2'] = '0.01*self.round'
    return algo_params
class FedKF_d1w0001_d2w001:
  Server=KFServer_d1w0001_d2w001
  Client=KFClient
