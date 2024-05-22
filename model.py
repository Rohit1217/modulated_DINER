import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class newMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
      super(newMLP, self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size ,bias=True)
      self.hash1=nn.parameter.Parameter(1e-4 * (torch.rand((hidden_size))*2 -1),requires_grad = True)
      self.fc2 = nn.Linear(hidden_size, hidden_size,bias=True)
      self.hash2=nn.parameter.Parameter(1e-4 * (torch.rand((hidden_size))*2 -1),requires_grad = True)
      self.fc3 = nn.Linear(hidden_size, hidden_size,bias=True)
      self.hash3=nn.parameter.Parameter(1e-4 * (torch.rand((hidden_size))*2 -1),requires_grad = True)
      self.fc4 = nn.Linear(hidden_size, output_size,bias=True)

    def forward(self,x,hash=True):
        x = F.gelu(self.fc1(x))+self.hash1
        x = F.gelu(self.fc2(x))+self.hash2
        x = F.gelu(self.fc3(x))+self.hash3
        x = self.fc4(x)
        #x = torch.clamp(x,-1.0,1.0)
        x = torch.clamp(x, min = -1.0,max = 1.0)
        return x
    
class hashLinear(nn.Module):
  def __init__(self,input_size,output_size):
    super(hashLinear,self).__init__()
    self.fc1=nn.Linear(input_size*2,output_size)
    self.hash1=nn.parameter.Parameter(1e-4 * (torch.rand((input_size))*2 -1),requires_grad = True)
  def forward(self,x):
    if len(x.size())<2:
      x=torch.cat((x,self.hash1))
      x=self.fc1(x)
    else:
      d=x.shape[0]
      y=torch.unsqueeze(self.hash1, 0)
      y = torch.repeat_interleave(y, d, dim=0)
      x=torch.cat((x,y),dim=1)
      x=self.fc1(x)
    return x

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        return out
    

class FineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=1):
        super().__init__()
        self.omega_0 = omega_0

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
                self.linear.bias.uniform_(-30,30)

    def forward(self, input):
        alpha=torch.abs(self.linear(input))+1
        out = torch.sin(self.omega_0 *alpha*( self.linear(input)))
        return out

class hashMLP(nn.Module):
  def __init__(self,input_size,hidden_size,output_size,inhash_size):
    super(hashMLP,self).__init__()
    self.fc1 = nn.Linear(input_size,inhash_size//2)
    self.fc2 = nn.Linear(inhash_size//2,inhash_size)
    self.inhash = nn.parameter.Parameter(1e-4 * (torch.rand((inhash_size))*2 -1),requires_grad = True)
    self.fc3 = nn.Linear(inhash_size,hidden_size//2)
    self.hashlin1 = hashLinear(hidden_size//2,hidden_size//2)
    self.hashlin2 = hashLinear(hidden_size//2,hidden_size//2)
    self.hashlin3 = hashLinear(hidden_size//2,hidden_size//2)
    self.hashlin4 = hashLinear(hidden_size//2,output_size)

  def forward(self,x):
    x = F.gelu(self.fc1(x))
    x = F.gelu(self.fc2(x))+self.inhash
    x = F.gelu(self.fc3(x))
    x = F.gelu(self.hashlin1(x))
    x = F.gelu(self.hashlin2(x))
    x = F.gelu(self.hashlin3(x))
    x = self.hashlin4(x)
    x = torch.clamp(x, min = -1.0,max = 1.0)
    return x

class DINERMlp(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,h,w):
        super(DINERMlp, self).__init__()
        h_size=1
        #self.register_parameter('hash', nn.Parameter((torch.randn((1200*1200,2))).to(Device),requires_grad=True))
        self.hash = nn.parameter.Parameter(1e-4 * (torch.rand((h*w,h_size))*2 -1),requires_grad = True)
        self.fc1 = nn.Linear(h_size, hidden_size ,bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size,bias=True)
        #self.fc4 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc5 = nn.Linear(hidden_size, output_size,bias=True)

    def forward(self,x,hash=True):
        hash3=self.hash
        if hash:
          #x=torch.cat((self.hash[:,0].unsqueeze(1),self.hash[:,0].unsqueeze(1).detach()),dim=1)
          x=hash3
        else:
          #x=torch.cat((x.unsqueeze(1),x.unsqueeze(1)),dim=1)
          x=x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = self.fc5(x)
        #x = torch.clamp(x,-1.0,1.0)
        x = torch.clamp(x, min = -1.0,max = 1.0)
        return x

class DinerSiren(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,h,w):
      super(DinerSiren, self).__init__()
      h_size=2
      #self.register_parameter('hash', nn.Parameter((torch.randn((1200*1200,2))).to(Device),requires_grad=True))
      self.hash = nn.parameter.Parameter(1e-4 * (torch.rand((h*w,h_size))*2 -1),requires_grad = True)
      self.fc1 = SineLayer(h_size, hidden_size ,bias=True,is_first=True)
      self.fc2 = SineLayer(hidden_size, hidden_size,bias=True)
      self.fc3 = SineLayer(hidden_size, hidden_size,bias=True)
      #self.fc4 = SineLayer(hidden_size, hidden_size,bias=True)
      self.fc5 = nn.Linear(hidden_size, output_size,bias=True)

    def forward(self,x,hash=True):
        hash3=self.hash
        if hash:
          x=hash3
        else:
          x=x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = F.relu(self.fc4(x))
        x = self.fc5(x)
        #x = torch.clamp(x,-1.0,1.0)
        x = torch.clamp(x, min = -1.0,max = 1.0)
        return x

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,h,w):
        super(MLP, self).__init__()
        self.hash1 = nn.parameter.Parameter(1e-4 * (torch.rand((h,15))*2 -1),requires_grad = True)
        self.hash2 = nn.parameter.Parameter(1e-4 * (torch.rand((15,w))*2 -1),requires_grad = True)
        self.fc1 = nn.Linear(input_size-1, hidden_size ,bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc4 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc5 = nn.Linear(hidden_size, output_size,bias=True)

    def forward(self,x,hash=True):
        hash3=torch.matmul(self.hash1,self.hash2)
        h,w=hash3.shape
        hash3=hash3.view(h*w,1)
        if hash:
          #x=torch.cat((self.hash[:,0].unsqueeze(1),self.hash[:,0].unsqueeze(1).detach()),dim=1)
          x=hash3
        else:
          #x=torch.cat((x.unsqueeze(1),x.unsqueeze(1)),dim=1)
          x=x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        #x = torch.clamp(x,-1.0,1.0)
        x = torch.clamp(x, min = -1.0,max = 1.0)
        return x

class hMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,h,w,d=20):
        super(hMLP, self).__init__()
        #self.register_parameter('hash', nn.Parameter((torch.randn((1200*1200,2))).to(Device),requires_grad=True))
        #self.hash = nn.parameter.Parameter(1e-4 * (torch.rand((h*w,2))*2 -1),requires_grad = True)
        self.d=d
        self.hash01 = nn.parameter.Parameter(1e-4 * (torch.rand((h,d))*2 -1),requires_grad = True)
        self.hash02 = nn.parameter.Parameter(1e-4 * (torch.rand((d,w))*2 -1),requires_grad = True)
        self.hash11 = nn.parameter.Parameter(1e-4 * (torch.rand((h,d))*2 -1),requires_grad = True)
        self.hash12 = nn.parameter.Parameter(1e-4 * (torch.rand((d,w))*2 -1),requires_grad = True)
        self.hash21 = nn.parameter.Parameter(1e-4 * (torch.rand((h,d))*2 -1),requires_grad = True)
        self.hash22 = nn.parameter.Parameter(1e-4 * (torch.rand((d,w))*2 -1),requires_grad = True)
        self.hash31 = nn.parameter.Parameter(1e-4 * (torch.rand((h,d))*2 -1),requires_grad = True)
        self.hash32 = nn.parameter.Parameter(1e-4 * (torch.rand((d,w))*2 -1),requires_grad = True)
        #self.hash41 = nn.parameter.Parameter(1e-4 * (torch.rand((h,d))*2 -1),requires_grad = True)
        #self.hash42 = nn.parameter.Parameter(1e-4 * (torch.rand((d,w))*2 -1),requires_grad = True)
        self.fc1 = nn.Linear(input_size-1, hidden_size ,bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size,bias=True)
        #self.fc4 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc5 = nn.Linear(hidden_size, output_size,bias=True)
        #self.fc6 = SineLayer(input_size-1, hidden_size,bias=True)
        #self.fc7 = SineLayer(input_size-1, hidden_size,bias=True)
        #self.fc8 = SineLayer(input_size-1, hidden_size,bias=True)
        #self.fc9 = SineLayer(input_size-1, hidden_size,bias=True)

    def forward(self,x,hash=True):
        hash1=torch.matmul(self.hash01,self.hash02)
        hash2=torch.matmul(self.hash11,self.hash12)
        hash3=torch.matmul(self.hash21,self.hash22)
        hash4=torch.matmul(self.hash31,self.hash32)
        #hash5=torch.matmul(self.hash41,self.hash42)
        h,w=hash1.shape
        hash1=hash1.view(h*w,1)
        hash2=hash2.view(h*w,1)
        hash3=hash3.view(h*w,1)
        hash4=hash4.view(h*w,1)
        #hash5=hash5.view(h*w,1)
        if hash:
          #x=torch.cat((self.hash[:,0].unsqueeze(1),self.hash[:,0].unsqueeze(1).detach()),dim=1)
          x=hash1
        else:
          #x=torch.cat((x.unsqueeze(1),x.unsqueeze(1)),dim=1)
          x=x

        x = F.relu(self.fc1(x))+ hash2
        x = F.relu(self.fc2(x))+ hash3
        x = F.relu(self.fc3(x))+ hash4
        #x = F.relu(self.fc4(x))+ hash5
        x = self.fc5(x)
        #x = torch.clamp(x,-1.0,1.0)
        x = torch.clamp(x, min = -1.0,max = 1.0)
        return x



class hsMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,h,w):
        super(hsMLP, self).__init__()
        d=20
        self.hash01 = nn.parameter.Parameter(1e-4 * (torch.rand((h,d))*2 -1),requires_grad = True)
        self.hash02 = nn.parameter.Parameter(1e-4 * (torch.rand((d,w))*2 -1),requires_grad = True)
        self.fc1 = nn.Linear(input_size-1, hidden_size ,bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc4 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc5 = nn.Linear(hidden_size, output_size,bias=True)
        self.fc6 = SineLayer(input_size-1, hidden_size,bias=True)
        self.fc7 = SineLayer(input_size-1, hidden_size,bias=True)
        self.fc8 = SineLayer(input_size-1, hidden_size,bias=True)
        self.fc9 = SineLayer(input_size-1, hidden_size,bias=True)

    def forward(self,x,hash=True):
        hash1=torch.matmul(self.hash01,self.hash02)
        h,w=hash1.shape
        hash1=hash1.view(h*w,1)
        if hash:
          x=hash1
        else:
          x=x

        x = F.relu(self.fc1(x))+(self.fc6((hash1)))
        x = F.relu(self.fc2(x))+(self.fc7((hash1)))
        x = F.relu(self.fc3(x))+(self.fc8((hash1)))
        x = F.relu(self.fc4(x))+(self.fc9((hash1)))
        x = self.fc5(x)
        x = torch.clamp(x, min = -1.0,max = 1.0)
        return x