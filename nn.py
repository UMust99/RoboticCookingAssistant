import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchvision.models as models
import PIL
from PIL import Image
#loading libraries 


# Load the directory paths to the dataset

IM_DIR = 'C:/Users/jozef/OneDrive/Desktop/FYP_images/Images'


valid_tfms = tt.Compose([tt.ToTensor()])
valid_ds = ImageFolder(IM_DIR, valid_tfms)

#trying to use GPU if available
torch.cuda.is_available()
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
device = get_default_device()



#define accuracy etc
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                      
        loss = F.cross_entropy(out, labels)                   # Calculate training loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                                    # Generate predictions
        loss = F.cross_entropy(out, labels)                   # Calculate validation loss
        acc = accuracy(out, labels)                           # Calculate accuracy
        return {'val_loss': loss.detach(),  'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()         # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()            # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.10f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))




def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels),     # Batch Normalization
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class CustomCNN(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 128)                                 # 3 x 64 x 64 
        self.conv2 = conv_block(128, 256, pool=True)                              # 128 x 32 x 32 
        self.res1 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))     # 256 x 32 x 32
        
        self.conv3 = conv_block(256, 512, pool=True)                              # 512 x 16 x 16
        self.conv4 = conv_block(512, 1024, pool=True)                             # 1024 x 8 x 8
        self.res2 = nn.Sequential(conv_block(1024, 1024), conv_block(1024, 1024)) # 1024 x 8 x 8

        self.conv5 = conv_block(1024, 2048, pool=True)                            # 256 x 8 x 8
        self.conv6 = conv_block(2048, 4096, pool=True)                            # 512 x 4 x 4
        self.res3 = nn.Sequential(conv_block(4096, 4096), conv_block(4096, 4096)) # 512 x 4 x 4
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),                          # 9216 x 1 x 1
                                        nn.Flatten(),                             # 9216
                                        nn.Linear(9216, num_classes))             # 131
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out    # Residual Block 
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out    # Residual Block
        out = self.classifier(out)
        return out



#define resnet model
class ResNetCNN(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)     # You can change the resnet model here
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 34)          # Output classes
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True





#load weights


input_channels = 3
output_classes = 34

custom_model = to_device(CustomCNN(input_channels, output_classes), device)
custom_model

custom_model.load_state_dict(torch.load('C:/Users/jozef/OneDrive/Desktop/FYP/Model2/ing-custom.pth', map_location=torch.device('cpu')))

custom_model.eval()



#resnet_model = to_device(ResNetCNN(), device)
#resnet_model

#resnet_model.load_state_dict(torch.load('C:/Users/jozef/OneDrive/Desktop/FYP/Model2/ing-resnet.pth', map_location=torch.device('cpu')))

#resnet_model.eval()


#function for predictions

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    classes = ['apple', 'avocado', 'bacon', 'banana', 'beef', 'bread', 'broccoli', 'carrot', 'cauliflower', 'celery', 'cheese', 'chicken', 'chili', 'cinnamon', 'corn', 'cucumber', 'egg', 'garlic', 'green_peas', 'ham', 'lemon', 'lettuce', 'mushroom', 'peach', 'pineapple', 'pork', 'potato', 'pumpkin', 'red_onion', 'red_pepper', 'salmon', 'sausage', 'strawberry', 'tomato']
    return classes[preds[0].item()]


with open('ingredients.txt', 'w') as f:
	for i in range(len(valid_ds)):
		img, label = valid_ds[i]
		pred = predict_image(img, custom_model)
		f.write(pred + ' ')

