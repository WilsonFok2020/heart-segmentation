# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 15:28:08 2018

@author: wfok007
"""
import torch, itertools

import numpy as np

from decompress import addAxis, fast_transfer
import os, time
from weightNorm import metaGen, Dice_Loss, non_differentiable_Dice_coef,  load_stat
from ensemblePrediction import create_Ensemble,  prepareInputsForGPU, ensemble_predict


if __name__ == '__main__':

    prog_dir = r'Z:\mpi_heart\ensembleTestPredictions'
    training_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training'
    output_dir = r'Z:\mpi_heart\howBigEnsemble'
    debug_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training\debug'
    hpc = r'Z:\mpi_heart'
    

    # setting up the environment for testing
    mean, std = load_stat()
    
    num_ranks = 125
    od, oh, ow = 88 , 112, 112
    multiple = 0.25
    minibatch = 5
    threshold = 0.37
    aa_mean, aa_std = addAxis(mean, std)
    gaa_mean, gaa_std = fast_transfer(aa_mean), fast_transfer(aa_std)
    

    # prepare the test set
    gen = metaGen('test', num_ranks, hpc)
    criterion = Dice_Loss()
    accuracy = non_differentiable_Dice_coef()
    
    # checkpoints to be tested
    weightNameList = [1480, 2440, 2880, 3320, 3920, 4440]
    # these are the filenames of the saved weights
    weightNameList = ['weights'+str(i*minibatch) for i in weightNameList]
    
    
    # combinations of different sizes of the ensemble (network using weights at different checkpoints)
    cpCombinations = {i:list(itertools.combinations(weightNameList,i)) for i in range(1,7)}
    
    MAX_EPOCH = 50 # number of test cases
    total_num_combinations = sum([len(cpCombinations[key]) for key in sorted(cpCombinations.keys())])
    print ('total_num_combinations %d' %total_num_combinations)
    
    #initialize an array to hold Dice coefficients
    coef_holder = np.zeros((total_num_combinations, MAX_EPOCH))
    
    counter = 0
    for key in sorted(cpCombinations.keys()):
        print (key)
        currentCombination = cpCombinations[key]
        print ('current combinations {}'.format(currentCombination))
        for groupMembers in currentCombination:
            
           
            ensemble = create_Ensemble(groupMembers)
            
            
            
            # get test cases ready
            start_time = time.time()
            for i, (full_image, full_mask, image, mask) in enumerate(gen):
                
                image1 , target = prepareInputsForGPU(image, full_mask, gaa_mean, gaa_std)
                
                with torch.no_grad():
                    # ensemble
                    output = ensemble_predict(ensemble, image1)
                coef2, true_pos_arr2, false_pos_arr2, false_neg_arr2 = accuracy(output, target, threshold=threshold, returnArr=True)
                print ('[%d] full resolution dice coef: %f' %(i, coef2))
                coef_holder[counter, i] = coef2
                if i >= MAX_EPOCH-1:
                    break
            
            del ensemble
            print ('time elasped: %f' %(time.time()-start_time))
            
            counter = counter + 1
    # save for plotting in the next step
    np.save(os.path.join(prog_dir, 'coef_holder2'), coef_holder)
##            
