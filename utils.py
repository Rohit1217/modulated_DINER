import torch
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np  

def disp_img_train(model,h,w):
  with torch.no_grad():
    genr_image=model(1)
    genr_image=genr_image.view(h,w,3)
    genr_image=genr_image*0.5+0.5
    plt.imshow(genr_image.to('cpu').detach().numpy())
    plt.show()


def PSNR(x,y):
  y=y*0.5+0.5
  loss=torch.pow(x-y,2)
  mse=torch.mean(loss)
  print(mse)
  psnr=-10*torch.log10(mse+1e-7)
  return psnr

def sum_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad) 

def save_model(model,optimizer,epoch):
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "model.pth")
  
def save_image(genr_image,path):
  image=genr_image
  image_np = image.to('cpu').detach().numpy() if isinstance(image, torch.Tensor) else image
  image_np = (image_np * 255).astype(np.uint8)
  pil_image = Image.fromarray(image_np)
  pil_image.save(path)

def imagefrom_hash(hash,h,w):
  with torch.no_grad():
    img=torch.zeros(h,w)
    for x in range(h):
      for y in range(w):
        img[x,y]=hash[w*x+y]
  return img

def get_svd_and_num_eigenvectors_above_threshold(matrix, threshold=0.5,Device='cpu'):
  U, S, V = torch.svd(matrix)
  matrix=torch.mm(torch.mm(U, torch.diag(S)), V.t())

  m = nn.Threshold(threshold,0)
  S=m(S)
  num_eigenvectors = torch.sum(S > threshold)
  new_matrix=torch.mm(torch.mm(U, torch.diag(S)), V.t()).to(Device)
  print(torch.dist(new_matrix,matrix.to(Device)))

  return new_matrix,num_eigenvectors,S

def residual_display(genr_image,new_img):
  diff_img=torch.abs(genr_image-new_img.permute(1,2,0))
  plt.imshow(diff_img.detach().to('cpu').numpy())
  plt.show()