# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 17:55:10 2018

@author: wfok007
"""


import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, itertools, random

def fileGen(tag, num_ranks, hpc):
    """
    This function creates a dataset python generator, an object that reads the 
    correct group of the datasets, shuffles the data within, returns
    samples one by one
    
    Inputs:
        tag = type of datasets
        num_ranks = number of files in which the dataset is split and stored
        hpc = directory
    
    Outputs:
        train_images = images
        train_ind = location of the ones in Images' masks
        th_images, th_ind = downsampled version of the same thing
    """
    # get a series of fileNames    
    names = [(tag+'2Rank_pairs'+str(i)+'.npz' ,tag+'2Rank_pairs'+str(i)+'_masks.p') for i in range(num_ranks)]
    
    # add randomness
    random.shuffle(names)
    

    for arrayName, pickleName in itertools.cycle(names):
        try:
            # catch I/O errors
            npzfile = np.load(os.path.join(hpc, arrayName))
            train_images = npzfile['images']
            th_images = npzfile['th_images']
            
        except OSError as err:
            print (err)
        try:
            train_ind, th_ind = pickle.load( open( os.path.join(hpc, pickleName), "rb" ) )
        except OSError as err:
            print (err)
        yield train_images, train_ind, th_images, th_ind

def decompressGen(train_images, train_ind):
    """
    This generator returns pair of stack of images and their mask one by one
    
    Inputs:
        train_images = images
        train_ind = location of the ones in Images' masks
    Output:
        a single case
    """
    
    num_samples = train_images.shape[0]
    assert (len(train_ind) == num_samples),"we have %d in index but %d in images" %(len(train_ind), train_images.shape[0])
    
    for i in range(num_samples):
        # change the file type
        image = train_images[i,...].astype(np.float32)
        mask = train_ind[i]
        yield image, mask

def metaGen(tag, num_ranks, hpc):
    """
    This generator contains file handling generator and decompression generator.
    
    Inputs:
        
        
        tag = type of datasets
        num_ranks = number of files in which the dataset is split and stored
        hpc = directory
    Outputs:
        full_image, full_mask = full resolution images and their masks
        image, mask = downsampled
    """
        
    for train_images, train_ind, th_images, th_ind in fileGen(tag, num_ranks, hpc):
        pack = zip(decompressGen(train_images, train_ind), decompressGen(th_images, th_ind))
        for (full_image, full_mask), (image, mask) in pack:
            yield full_image, full_mask, image, mask

def freeze_layer(layer):
    """
    The function stops keeping track of the gradients for the list of layers.
    It eliminates the storage of error gradients for some layers
    Inputs:
        layer = layer that needs to be frozen
    """
    for param in layer.parameters():
        param.requires_grad = False


class interpolate3D(nn.Module):
    """
    This object can interpolate feature cubes in 3D.
    
    """
    def __init__(self, od, oh, ow):
        super().__init__()
        # create a grid of the right dimensions (od, oh ,ow)
        wgrid = np.zeros((od,oh,ow,3))
        # Pytorch uses -1 - 1
        x_start = -1
        x_end = 1
        x_step = ow
        # no need to specify type as this line runs on the cpu
        a = np.linspace(x_start, x_end, num=x_step)
        
        y_start = -1
        y_end = 1
        y_step = oh
        b = np.linspace(y_start, y_end, num=y_step)
        
        z_start = -1
        z_end = 1
        z_step = od
        c = np.linspace(z_start, z_end, num=z_step)
        
        for z in range(od):
            for x in range(ow):
                for y in range(oh):
                    # the fix is here.
                    #the order in the last dimension is in complete opposite to the order of the documentation !
                    wgrid[z,y,x,0] = a[x] 
                    wgrid[z,y,x,1] = b[y]
                    wgrid[z,y,x,2] = c[z]
        
        # minibatch size
        wgrid = wgrid[np.newaxis,...]
        
        grid = torch.from_numpy(wgrid.astype(np.float32))
        # save the grid in the right type (GPU) for later use
        self.grid = grid.type(torch.cuda.FloatTensor)
    
        
    def _check_input_dim(self, input):
        # check 1,C,D,H,W
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        
    def forward(self, x):
        self._check_input_dim(x)
        # interpolation
        return F.grid_sample(x, self.grid)

if __name__ == '__main__':
    pass