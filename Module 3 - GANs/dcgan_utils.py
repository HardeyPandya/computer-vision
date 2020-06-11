# Deep Convolutional GANs

# Importing the libraries
import torch
import torch.nn as nn
import torch.nn.parallel

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the Generator

class G(nn.Module): #inherit neural net module from torch.nn
    def __init__(self):
        super(G, self).__init__() #to activate the inheritance
        self.main = nn.Sequential(#meta module which is composed of many different modules
            #Inverse Vonvolution
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),#(inputsize,n_featuremaps, kernel_sizes, strides, padding, bias)
            #Bacth normalization
            nn.BatchNorm2d(512),
            #Rectifier
            nn.ReLU(True),
            
            #Inverse Convolution
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            #Inverse Convolution
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            #Inverse Convolution
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            #Inverse Convolution
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):#input is random vector of size 100
        output = self.main(input)
        return output

# Discriminator
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            #1st Convolutional Layer
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            #Rectifier leaky
            nn.LeakyReLU(0.2, inplace=True),
            
            #2nd Convolutional Layer
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            #3rd Convolutional Layer
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            #4th Convolutional Layer
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            #5th Convolutional Layer
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1) #flattening results into 1 dimension
