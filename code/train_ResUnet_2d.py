import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import os
import imageio
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from data import *
# from newdata import *

from torch import multiprocessing, cuda
from torch.backends import cudnn
# from unet3D import *
from metric import *

import torchvision.models
from torchvision.utils import save_image
from PIL import Image

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

from torch.utils.data import Subset
import numpy as np
import math

from ResUnet_2d import *

# import gc

#Set device
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

dtypeLong = torch.cuda.LongTensor 
torch.cuda.set_device(0)

#Hyperparameters
in_channel = 4
num_classes = 4
lr = 0.001
batch_size = 1
num_epochs = 20
load_model = True

saved_model_path = "Resunet_2D_overfit.pth.tar"

my_transforms = transforms.Compose([
    transforms.ToTensor()
])


dataset = BRATS_Dataset(csv_file = '../data/HGG_CSV.csv', root_dir = '../data/HGG/', transform = my_transforms)

# dataset = BRATS_Dataset(csv_file = '../dummy_data/dummy_data.csv', root_dir = '../dummy_data/HGG/', transform = my_transforms)

#Dividing the dataset into two parts
train_set, test_set = torch.utils.data.random_split(dataset, [200, 59])

# train_set, test_set = torch.utils.data.random_split(dataset, [1, 1])


train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

torch.cuda.empty_cache()

#Initialize Network
net = ResUnet()
model = net.cuda()



#Saving the model after every 2 epochs
def save_checkpoint(checkpoint,filename = saved_model_path):
    torch.save(checkpoint,filename)

#Load the saved model
def load_checkpoint(checkpoint):
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

if load_model:
    load_checkpoint(torch.load(saved_model_path))

seed = 42

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True

###########   LOSS & OPTIMIZER   ##########

weights = torch.Tensor([0.05, 0.95, 0.95, 0.95])
weights = weights.cuda()
momentum = 0.9
step_size = 15
gamma = 0.5


bce = nn.BCELoss(weight=weights)

optimizer = optim.SGD(model.parameters(),lr=lr, momentum=momentum, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)        
        
patch_size = 8

# torch.autograd.set_detect_anomaly(True)

#Train the network
for epoch in range(1,num_epochs+1):
    
    total_loss = 0

    if epoch%1==0:
        checkpoint = {'state_dict':net.state_dict(),'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)
#         model.eval()
    scheduler.step()
    print('Learning rate= '+str(optimizer.param_groups[0]['lr']))
    batch_loss = 0
    no_of_batch = 0
    
    
    for batch_idx, (image, mask) in tqdm(enumerate(train_loader)):
        
        batch_loss = 0
        no_of_patch = 0
        no_of_batch = no_of_batch +1
        
        model.train()
        
        image = image.cuda()
        image = image.type(dtype)
        mask = mask.cuda()
        mask = mask.type(dtype)


        for i in range(6,105,patch_size):
            no_of_patch = no_of_patch + 1
            image_new = image[:,2,:,:,i:i+patch_size]
            mask_new = mask[:, :, :, :, i:i+patch_size]
            
            image_new = image_new.permute(3,0,1,2)
            mask_new = mask_new.squeeze(0).permute(3,0,1,2)
            
            
            predicted_mask = model(image_new)
            optimizer.zero_grad()
            
            
            
            dice_coeff,dice_coeff1,dice_coeff2,dice_coeff3 = multi_class_dice3(mask_new.cuda(),predicted_mask.cuda())
            
            
            print("Total Dice coeff "+str(dice_coeff)+" at epoch "+str(epoch))
            
            loss = 1 - dice_coeff
            
            batch_loss += dice_coeff
#             total_loss += dice_coeff
            loss.backward()
            optimizer.step()
        
        total_loss += batch_loss.item()/no_of_patch
        print("Dice coeff. at batch "+str(batch_idx)+" = "+str(batch_loss.item()/no_of_patch)) 
            
    print("Dice coeff. at epoch "+str(epoch)+" = "+str(total_loss/no_of_batch)) 
        


def calculate_loss(loader):
    batch_loss = 0
    patch_size = 8
    no_of_batch = 0
    total_loss = 0
    total_dice = 0
    total_dice0 = 0
    total_dice1 = 0
    total_dice2 = 0
    total_dice3 = 0
    
    with torch.no_grad():
        for batch_idx, (image, mask) in tqdm(enumerate(loader)):
            batch_loss = 0
            no_of_patch = 0
            no_of_batch = no_of_batch +1

            model.eval()

            image = image.cuda()
            image = image.type(dtype)
            mask = mask.cuda()
            mask = mask.type(dtype)
            
#             final_predicted = torch.randn(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],mask.shape[4])
#             final_predicted[:,:,:,:,0:6] = mask[:,:,:,:,0:6]

            for i in range(6,105,patch_size):
                no_of_patch = no_of_patch + 1
                image_new = image[:,2,:,:,i:i+patch_size]
                mask_new = mask[:, :, :, :, i:i+patch_size]
                
                image_new = image_new.permute(3,0,1,2)
                mask_new = mask_new.squeeze(0).permute(3,0,1,2)


                predicted_mask = model(image_new)
                
#                 final_predicted[:,:,:,:,i:i+patch_size] = predicted_mask
                
                
                
                dice_coeff,dice_coeff1,dice_coeff2,dice_coeff3 = multi_class_dice3(mask_new.cuda(),predicted_mask.cuda())
            
            
                print("Dice loss for tumor 1 image "+str(dice_coeff1)+" batch number "+str(no_of_batch))
                print("Dice loss for tumor 2 image "+str(dice_coeff2)+" batch number "+str(no_of_batch))
                print("Dice loss for tumor 3 image "+str(dice_coeff3)+" batch number "+str(no_of_batch))
                print("Total Dice loss "+str(dice_coeff)+" batch number "+str(no_of_batch))

                total_dice+=dice_coeff
                total_dice1+=dice_coeff1
                total_dice2+=dice_coeff2
                total_dice3+=dice_coeff3
                

#             final_predicted[:,:,:,:,134:] = mask[:,:,:,:,134:]
            
            
            
            
        print("Final Dice loss for tumor 1 image "+str(total_dice1/(no_of_batch*16)))
        print("Final Dice loss for tumor 2 image "+str(total_dice2/(no_of_batch*16)))
        print("Final Dice loss for tumor 3 image "+str(total_dice3/(no_of_batch*16)))
        print("Final Total Dice loss "+str(total_dice/(no_of_batch*16)))




print("Accuracy on Train Set")
calculate_loss(train_loader)  
print("Accuracy on Test Set")
calculate_loss(test_loader)  
    