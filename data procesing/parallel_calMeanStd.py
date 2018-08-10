# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 22:15:46 2018

@author: wfok007
"""

from mpi4py import MPI
import os
import numpy as np

def rescale(x, vmax=255):
    """
    rescaling
    """
    return x/ vmax 

if __name__ == '__main__':
    
    training_dir = r'/hpc/wfok007/mpi_heart'
    
    # dataset saved across this number of numpy arrays
    num_images_arrays = 125
    cpus = 25 # using 25 cpus
    # Because we have much training data, repeat the process a few times
    repeats = int(num_images_arrays/cpus)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tag = 'train' # work on training data
    
    for i in range(repeats):
        base_rank = i*cpus
          
        r = base_rank + rank # update IDs
            
        names = (tag+'2Rank_pairs'+str(r)+'.npz' ,tag+'2Rank_pairs'+str(r)+'_masks.p')
        
        arrayName = names[0]
        npzfile = np.load(os.path.join(training_dir, arrayName))
        train_images = rescale(npzfile['images'])
        
        depth = train_images.shape[1]
        print ('I am rank %d; depth %d' %(r,depth))
        
        # calculate mean and standard deviation
        mean = np.asarray([np.mean(train_images[:,i,:,:]) for i in range(depth)])
        
        std = np.asarray([np.std(train_images[:,i,:,:]) for i in range(depth)])
        # calculate the mean and standard deviation of the intensity
        np.savez(os.path.join(training_dir, 'kernel_mean_std_'+str(r)), mean=mean, std=std)