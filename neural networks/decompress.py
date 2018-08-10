# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:36:23 2018

@author: wfok007
"""

import torch
import numpy as np

def rescale(x, vmax=255):
    return x/ vmax

def normalise(image, mask, mean, std):
    '''     
    This function normalizes the input/ stack of images depth-wise
    
    Inputs:
        image, mask = images and their masks
        input float32, range 0-255 or 0-1
        dimension: depth, channels, height, width
        
        mean, std = input statistics (1D numpy array)
    
   
    Outputs:
        images and masks = float32
    '''
    #normalization 0-255 -> 0-1
    image = rescale(image)
    image = (image - mean[:,np.newaxis, np.newaxis, np.newaxis])/ std[:,np.newaxis, np.newaxis, np.newaxis]
    
    # careful up-casting
    return image.astype(np.float32), mask[np.newaxis, np.newaxis,...].astype(np.float32)

def addAxis(mean, std):
    """
    add new dimensions to inputs
    Inputs:
        mean and standard derivation 
    
    Outputs:
        mean and standard derivation in float32
    
    """
    mean = mean[:,np.newaxis, np.newaxis, np.newaxis]
    std = std[:,np.newaxis, np.newaxis, np.newaxis]
    return mean.astype(np.float32), std.astype(np.float32)  

def gpuNormalise(gImage, gMask, gMean, gStd):
    ''' just like the normalize but on the gpu in lieu of cpu
    
    Inputs:
        gImage, gMask = images and masks on GPU
        gMean, gStd = statistics on GPU
    
    Outputs:
        normalized images and masks
    
    '''
    gImage = rescale(gImage)
    gImage = (gImage - gMean)/ gStd
    
    gMask = gMask.unsqueeze(0)
    gMask = gMask.unsqueeze(0)
    return gImage, gMask
    
def gpuDecompress(sc_img):
    ''' This function decompresses the color channels on GPU 
    Input:
        sc_img = images
    Output:
        images with 3 color channels
    
    '''
    sc_img = sc_img.unsqueeze(1)
    sc_img = sc_img.expand(-1, 3, -1, -1)
    return sc_img

def gpuDecompressMask(ind, d, h, w):
    """
    This function puts mask on the GPU
    Inputs:
        ind = where 1s are
        d,h,w = dimension of the mask
    Outputs
        mask = GPU mask
    """
    
    #new_mask = torch.zeros(d*h*w) does not sit on the gpu
    new_mask = torch.cuda.FloatTensor(d*h*w).fill_(0)
    new_mask[ind] = 1
    new_mask = new_mask.view(d,h,w)
    return new_mask

def decompress_mask(ind, d, h, w):
    """
    This function decompresses mask on CPU.
    Inputs:
        ind = where 1s are
        d,h,w = dimension of the mask
    Outputs
        mask = CPU mask
    """
    
    new_mask = np.zeros((d*h*w), dtype=np.uint8)
    new_mask[ind] = 1 # or 255 depends on how we normalize
    new_mask = new_mask.reshape(d,h,w)
    return new_mask.astype(np.float32)

def fast_transfer(x, isImage=True):
    """
    This function follows online tutorial on how to push a numpy array to torch array on GPU
    Inputs:
        x = either a stack of images or masks
        isImage = the identity of x
    Outputs
        x on GPU
        
    """
    if isImage:
        x = torch.from_numpy(x.astype(np.float32))
    else:
       x = torch.from_numpy(x.astype(np.int64)) 
    x = x.pin_memory()
    return x.cuda(non_blocking=True)
    
if __name__ == '__main__':
    pass
