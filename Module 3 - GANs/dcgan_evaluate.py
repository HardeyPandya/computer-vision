# Deep Convolutional GANs

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.parallel
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.autograd import Variable
from dcgan_utils import *
import torch.optim as optim

# Setting some hyperparameters
batchSize = 16 # We set the size of the batch.
imageSize = 64 # We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 
    # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

#Loading state if present
try:
    state = torch.load('results/state_last.torch')
    
    #Creating the nets
    netG = G()
    netG.load_state_dict(state["G"]) 
                
    netD = D()
    netD.load_state_dict(state["D"])
    
except:
    print("No state file found")
    exit()

#Testing
print("Testing")
noise = torch.randn(batchSize, 100, 1, 1)
noise = Variable(noise)
fake = netG(noise) #minibatch of fake images

vutils.save_image(fake.data,  '%s/evaluation_epoch_%03d_org.png' % ("./results", state["epoch"]))

#Normalize

fake = fake.view(fake.size(1), -1)
fake -= fake.min(1, keepdim=True)[0]
fake /= fake.max(1, keepdim=True)[0]
fake = fake.view(batchSize, 3, 64, 64)

#Saving
vutils.save_image(fake.data,  '%s/evaluation_epoch_%03d_norm.png' % ("./results", state["epoch"]))





