# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:56:12 2018

@author: wfok007
"""

import os, random, math
import numpy as np

def splitTrainTestSet(images, masks, trainPercentage, validationPercentage, testPercentage):
    '''
    This function randomly splits dataset into 3 groups (training / validation/ testing)
    Inputs: 
    images: MRI image stacks
    masks: their masks
    trainPercentage: percentage used for training
    validationPercentage: percentage used for validation
    testPercentage: percentage used for testing
        
    Outputs:
    train_images : images for training
    valid_images : images for validation
    test_images : images for testing
    train_masks: their masks for training
    valid_masks : their masks for validation
    test_masks : their masks for testing
    
    '''
    
    num_samples = images.shape[0]
    
    # split and divide
    # round to the nearest integer
    trainSize = math.floor(trainPercentage*num_samples)
    validationSize = math.floor(validationPercentage*num_samples)
    testSize = math.floor(testPercentage*num_samples)
    # shuffle the indices
    index = [i for i in range(num_samples)]
    random.shuffle(index)
    # allocate chunks of indices to different groups
    train_ind = index[0:trainSize]
    valid_ind = index[trainSize:trainSize+validationSize]
    test_ind = index[-testSize::]
    
    train_images = images[train_ind,...]
    valid_images = images[valid_ind,...]
    test_images = images[test_ind,...]
    train_masks = masks[train_ind,...]
    valid_masks = masks[valid_ind,...]
    test_masks = masks[test_ind, ...]
    
    # check the shape of the datasets
    for j, who in zip([train_images, valid_images, test_images, train_masks, valid_masks, test_masks],
                 ['train_images', 'valid_images', 'test_images', 'train_masks', 'valid_masks', 'test_masks']):
        print ('{} : {}'.format(who, j.shape))
        
    return train_images, valid_images, test_images, train_masks, valid_masks, test_masks

if __name__ == '__main__':

    training_dir = r'/hpc/wfok007/mpi_heart'
    fileName = 'input_data.npz'
    npzfile = np.load(os.path.join(training_dir, fileName))
    images = npzfile['images']
    targets = npzfile['masks']
    
    print ('image shape {}'.format(images.shape))
    print ('target shape {}'.format(targets.shape))
    

#                
    temp = splitTrainTestSet(images,targets, 0.70, 0.1, 0.2)
    
    train_images, valid_images, test_images, train_targets, valid_targets, test_targets = temp

    fileName = 'split2'
    np.savez_compressed(os.path.join(training_dir, fileName),
                        train_images=train_images,
                        valid_images=valid_images,
                        test_images=test_images,
                        train_targets=train_targets,
                        valid_targets=valid_targets,
                        test_targets=test_targets)
    