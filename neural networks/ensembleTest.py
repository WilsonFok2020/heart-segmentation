# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 18:00:08 2018

@author: wfok007
"""

from __future__ import print_function, division

import torch, math

import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from decompress import addAxis, fast_transfer
import os
from weightNorm import metaGen, Dice_Loss, non_differentiable_Dice_coef,  load_stat
from ensemblePrediction import create_Ensemble,  prepareInputsForGPU, ensemble_predict

if __name__ == '__main__':
    
    plt.close('all')
    
    prog_dir = r'Z:\mpi_heart\ensembleTestPredictions'
    training_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training'
    output_dir = r'Z:\mpi_heart\howBigEnsemble'
    debug_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training\debug'
    hpc = r'Z:\mpi_heart'
    weight_dir = 'Z:\mpi_heart\weight_checkpoints'
    

    
    mean, std = load_stat()
    
    num_ranks = 125
    od, oh, ow = 88 , 112, 112
    multiple = 0.25
    minibatch = 5
    threshold = 0.37
    aa_mean, aa_std = addAxis(mean, std)
    gaa_mean, gaa_std = fast_transfer(aa_mean), fast_transfer(aa_std)
    

    
    gen = metaGen('test', num_ranks, hpc)
    criterion = Dice_Loss()
    accuracy = non_differentiable_Dice_coef()
    
    chosen_num_members = 4
    weightNameList = [1480, 2440, 2880, 3320, 3920, 4440]
    weightNameList = ['weights'+str(i*minibatch) for i in weightNameList]
    
    # update the list
    groupMembers = weightNameList[0:chosen_num_members]
    ensemble = create_Ensemble(groupMembers)
    
    coefs2 = []
    MAX_EPOCH = 200 # the number of test cases
    
    for i, (full_image, full_mask, image, mask) in enumerate(gen):
        image1, target = prepareInputsForGPU(image, full_mask,gaa_mean, gaa_std, 88, 576, 576)
        
        with torch.no_grad():
            # ensemble
            output = ensemble_predict(ensemble, image1)
            
        coef2, true_pos_arr2, false_pos_arr2, false_neg_arr2 = accuracy(output, target, threshold=threshold, returnArr=True)
        print ('[%d] full resolution dice coef: %f' %(i, coef2))
        coefs2.append(coef2)
        
        
        sig = nn.functional.sigmoid(output)
        sig = sig.cpu().detach().numpy()
        pred_mask = sig[0,0,...]
        
       
        del output, sig, image1, target
        
        # save and draw the predictions and mistakes
        
#        pred_mask = pred_mask >= threshold
#        pred_mask = pred_mask *1
#        
#        
#    
#        temp_img = fromFloat_to_uint8(full_image)
#        
#        
#        tag = str(i)+'_'+ str(round(coef2, 3)).replace('.', 'p')
#        
#        save(temp_img, pred_mask, 'test_'+ tag, prog_dir)
#        
#        # need to decompress the index to get mask
#        full_mask = decompress_mask(full_mask,88, 576, 576)
#        
#        save(temp_img, full_mask, 'test_groundTruth_'+tag, prog_dir)
#        
#        # don't put segment of codes far apart !!!
#        true_pos_arr2 = true_pos_arr2[0,0,...]
#        false_pos_arr2 = false_pos_arr2[0,0,...]
#        false_neg_arr2 = false_neg_arr2[0,0,...]
#        save(temp_img, true_pos_arr2, 'test_true_pos_'+tag, prog_dir)
#        save(temp_img, false_pos_arr2, 'test_false_pos_'+tag, prog_dir)
#        save(temp_img, false_neg_arr2, 'test_false_neg_'+tag, prog_dir)
        
        
        if i >= MAX_EPOCH-1:
            break
    print ('finish testing')
    print ('median full resolution dice coefficient %f' %(np.median(coefs2)))
    print ('std full resolution dice coefficient %f' %(np.std(coefs2)))
    print ('maximum allocated %f GB' %(torch.cuda.max_memory_allocated()/math.pow(10,9)))
        