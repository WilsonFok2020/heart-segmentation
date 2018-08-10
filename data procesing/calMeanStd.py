# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:17:49 2018

@author: wfok007
"""

import os, time
import numpy as np
from parallel_calMeanStd import rescale
           
if __name__ == '__main__':

    training_dir = r'/hpc/wfok007/mpi_heart'
    
    num_ranks = 125
    tag = 'train'

    names = (tag+'2Rank_pairs'+str(0)+'.npz' ,tag+'2Rank_pairs'+str(0)+'_masks.p')
    print (names)
    
    arrayName = names[0]
    npzfile = np.load(os.path.join(training_dir, arrayName))
    train_images = rescale(npzfile['images'])
    
    depth = train_images.shape[1]
    
    # initialize array to hold the statistics
    means = np.zeros((depth,))
    stds = np.zeros((depth,))
    
    names = ['kernel_mean_std_'+str(i)+'.npz' for i in range(num_ranks)]
    for name in names:
        npzfile = np.load(os.path.join(training_dir, name))
        mean = npzfile['mean']
        std = npzfile['std']
        
        means += mean
        stds += std
    
    # calculate the aggregate
    means /= num_ranks
    stds /= num_ranks
    np.savez(os.path.join(training_dir, 'kernel_overall_mean_std'), mean=means, std=stds)
        
#    
#    
    
        
#    
   