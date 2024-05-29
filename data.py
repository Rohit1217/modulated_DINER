import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image



def normalize_image(image_path):
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    transform1 = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5,0.5], std=[0.5, 0.5, 0.5,0.5])
    ])
    transform2 = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image=transform(image)
    if image.shape[0]>3:
     normalized_image = transform1(image)
     return normalized_image[:3,:,:]
    else:
      normalized_image = transform2(image)
      return normalized_image

class CustomDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path
        self.normalized_image = normalize_image(self.image_path)
        self.alpha=10

    def __len__(self):
        len=self.normalized_image.size(1)*self.normalized_image.size(2)
        return len

    def __getitem__(self, idx):
        x,y=idx//self.normalized_image.size(2),idx%self.normalized_image.size(2)
        #print(x,y)
        coordinates=torch.tensor([x,y],dtype=torch.float32)
        rgb_values = self.normalized_image[:,x,y] # Replace with actual RGB values
        return coordinates,rgb_values
    

def get_coord_rgb(dataset):
   coord_x=torch.zeros(len(dataset),2)
   rgb_values_x=torch.zeros(len(dataset),3)
   for i in range(len(dataset)):
     coord_x[i]=dataset[i][0]
     rgb_values_x[i]=dataset[i][1]
   return coord_x,rgb_values_x
#print(coord_x,rgb_values_x)