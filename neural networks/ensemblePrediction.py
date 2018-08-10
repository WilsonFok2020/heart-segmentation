# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:23:52 2018

@author: wfok007
"""

from __future__ import print_function, division

import torch, math, os

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from weightNorm import createModel, metaGen, Dice_Loss, non_differentiable_Dice_coef, prepareInputsForGPU,  load_stat
from decompress import addAxis, fast_transfer, decompress_mask


def ensemble_predict(ensemble, image1):
    '''
    inputs:
        ensemble: dictionary that holds all the constituents
        image1: input image
    
    outputs:
        averaged prediction
    '''
    num_constituents = len(ensemble.keys())
    for i, key in enumerate(ensemble.keys()):
        modelA = ensemble[key]
        if i ==0:
            
            output = modelA(image1)
        else:
            output += modelA(image1)
    
    # take a simple average
    output /= num_constituents
            
    
    return output
def inference(ensemble, num_samples, gaa_mean, gaa_std, generator_testing, criterion, od, oh, ow):
    """
    This function performs inference of the network on the GPU using any dataset.
    
    Inputs:
        ensemble = dictionary of individual networks
        num_samples = how many cases we want to infer
        gaa_mean, gaa_std = input statistics
        generator_testing = dataset as a python generator object
        criterion = Dice loss
        od, oh, ow = depth, height , width
    
    Outputs:
        pred_mask = the predictions made by the ensemble
        ground_truth = target masks
        losses = Dice loss for each prediction
    
    """
        
    # initialize array to hold the results
    pred_mask = np.zeros((num_samples, od, oh, ow), dtype=np.float32)
    ground_truth = np.zeros((num_samples, od, oh, ow), dtype=np.uint8)
    losses = np.zeros((num_samples,), dtype=np.float32)
    
    for i, (full_image, full_mask, image, mask) in enumerate(generator_testing):
        
        image1, target = prepareInputsForGPU(image, full_mask,gaa_mean, gaa_std, od, oh, ow)
        # CPU only, just for storage
        mask = decompress_mask(full_mask, od, oh, ow)
        with torch.no_grad():
            # ensemble
            output = ensemble_predict(ensemble, image1)
            loss = criterion(output, target)
            sig = nn.functional.sigmoid(output)
            sig = sig.cpu().detach().numpy()
            pred_mask[i,...] = sig[0,0,...]
            ground_truth[i,...] = mask.astype(np.uint8)
            
        del output, sig, image1, target
        losses[i] = loss.item()
        print ('[%d] loss %f' %(i, losses[i]))
        
        if i == num_samples-1:
            break
        
    return pred_mask, ground_truth, losses

def create_Ensemble(weightNameList,
                    weight_dir =r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training\fasterLoadingWeights',
                    od=88, oh=112,ow=112, multiple=0.25):
    '''
    input:
        weightName List is a list of the weights for the models at different checkpoints
    output:
        dictionary of models
    '''
    ensemble = {}
    print ('creating ensemble with %d members' %(len(weightNameList)))
    for i, name in enumerate(weightNameList):
         modelA = createModel(od, oh,ow, multiple)
         print('loading %s' %name)
         modelA.load_state_dict(torch.load(os.path.join(weight_dir, name)))
         ensemble[i] = modelA
    
    print ('ensemble created')
    return ensemble
         
         
if __name__ == '__main__':
    
    plt.close('all')
    
    prog_dir = r'Z:\mpi_heart\optimisedDiceBars'
    training_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training'
    output_dir = r'Z:\mpi_heart\howBigEnsemble'
    debug_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training\debug'
    hpc = r'Z:\mpi_heart'
    weight_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training\fasterLoadingWeights'
    

    
    mean, std = load_stat()
    
    num_ranks = 125
    od, oh, ow = 88 , 112, 112
    multiple = 0.25
    minibatch = 5
    aa_mean, aa_std = addAxis(mean, std)
    gaa_mean, gaa_std = fast_transfer(aa_mean), fast_transfer(aa_std)
    

    
    gen = metaGen('valid', num_ranks, hpc)
    criterion = Dice_Loss()
    accuracy = non_differentiable_Dice_coef()
    
    # chosen checkpoints of network's weights
    weightNameList = [1480, 2440, 2880, 3320, 3920, 4440]
    weightNameList = ['weights'+str(i*minibatch) for i in weightNameList]
    num_checks = 200
    
    # determine the optimal number of members in the ensemble
    for i in range(1,len(weightNameList)+1):
        groupMembers = weightNameList[0:i]
        print (groupMembers)
        # loop through to see how many networks are useful
        ensemble = create_Ensemble(groupMembers)
        pred_mask, ground_truth, losses = inference(ensemble, num_checks, gaa_mean, gaa_std, gen, criterion, 88, 576, 576)
        
        fileName = str(i) +'_' +'valid_ensemble_loss_predict_mask'
            
            
        print ('Loss %f' %np.mean(losses)) 
        print ('[%d] maximum allocated %f GB' %(i, torch.cuda.max_memory_allocated()/math.pow(10,9)))
        np.savez_compressed(os.path.join(output_dir, fileName),
                            losses=losses)
        # running out of ram
        del pred_mask, ground_truth, losses, ensemble
        # predictions and ground truth are huge, but needed for later
#        np.savez_compressed(os.path.join(prog_dir, fileName),
#                            pred_mask=pred_mask,
#                            ground_truth=ground_truth,
#                            losses=losses)

 