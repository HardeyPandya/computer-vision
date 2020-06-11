# Deep Convolutional GANs

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.autograd import Variable
from dcgan_utils import *
import torch.optim as optim

# Setting some hyperparameters
batchSize = 64 # We set the size of the batch.
imageSize = 64 # We set the size of the generated images (64x64).
maxEpoch = 25

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 
    # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.

#Loading state if present
try:
    state = torch.load('/results/state_last.torch')
    print("State loaded [%d/%d][%d/%d]" % (state["epoch"], maxEpoch, state["minibatch"], len(dataloader)) )
    
    #Creating the nets
    netG = G()
    netG.load_state_dict(state["G"]) 
                
    netD = D()
    netD.load_state_dict(state["D"])
    
    #Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr = 0.002, betas = (0.5,0.999))
    optimizerG.load_state_dict(state['optimizerG'])
    
    optimizerD = optim.Adam(netD.parameters(), lr = 0.002, betas = (0.5,0.999)) 
    optimizerD.load_state_dict(state['optimizerD'])
    
except:
    print("No state file found")
    print("State created")
    
    state = {
        'epoch': 0,
        'minibatch': 0,
    }
    
    #Creating the nets
    netG = G()
    netG.apply(weights_init)
                
    netD = D()
    netD.apply(weights_init)

    #Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr = 0.002, betas = (0.5,0.999))
    
    optimizerD = optim.Adam(netD.parameters(), lr = 0.002, betas = (0.5,0.999)) 
print(state)

#Loss Function
criterion = nn.BCELoss()


def main():
    print()

if __name__ == '__main__':
    
    for epoch in range(state["epoch"], maxEpoch): #larger number can be tested
        #Each epoch will go trough all the images of the dataset
        
        for i, data in enumerate(dataloader, 0):
            #for minibatch in dataset
            #Data is a minibatch containing images and labels
            
            ###Update the weights of the Discriminator
            #Initialize gradients with respect to the weights
            netD.zero_grad()
            
            ##Training the discriminator with real image
            #Data Setup
            real, _ = data
            input = Variable(real)
            target = Variable(torch.ones(input.size()[0])) #creates tensor of ones
                    # input.size()[0] = batch size = 64
            #Forward pass of real data on Discriminator
            output = netD(input)
            #Error calculation
            errD_real = criterion(output, target)
            
            ##Training the discriminator with fake image
            #Data setup
            noise = torch.randn(input.size()[0], 100, 1, 1)
            noise = Variable(noise)
            fake = netG(noise) #minibatch of fake images
                    #torch Variable containing: tensor of predictions and gradients
            target = Variable(torch.zeros(input.size()[0])) #creates tensor of ones
            #Forward pass of generated fakes on Discriminator
            output = netD(fake.detach()) #apply forward pass only on the tensor of predictions to save memory
            #Error calculation
            errD_fake = criterion(output, target)
            
            #Backpropagation
            errD = errD_real + errD_fake
            errD.backward() #computes the gradient of the tensor of errors
            optimizerD.step()
            
            
            ###Update the weights of the Generator
            #Gradient initialization
            netG.zero_grad()
            #Data setup
            target = Variable(torch.ones(input.size()[0]))
            #Forward pass of generated fakes on Discriminator
            output = netD(fake) #gradient is kept
            #Error calculation
            errG = criterion(output, target)
            #Backpropagation
            errG.backward()
            optimizerG.step()
            
            ###Printing and saving
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, maxEpoch, i, len(dataloader), errD.data[0], errG.data[0]))
            if i%100 == 0: #every 100 steps or minibatches
                print("Saving")
                #Saving images
                vutils.save_image(real, '%s/real_samples.png' % "./results")
                fake = netG(noise)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch))
                
                ###Saving everything
                state = {
                    'epoch': epoch,
                    'minibatch': i,
                    'G': netG.state_dict(),
                    'D': netD.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    'optimizerG': optimizerG.state_dict()
                }
                torch.save(state, '%s/state_epoch_%03d.torch' % ("./results", epoch))
                torch.save(state, '%s/state_last.torch' % ("./results"))                        
    print("Training complete")
    
#if __name__ == '__main__':
#    main()

#Testing
print("Testing")
noise = torch.randn(64, 100, 1, 1)
noise = Variable(noise)
fake = netG(noise) #minibatch of fake images
vutils.save_image(fake.data, '%s/test.png' % ("./results"))

                








