# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:29:50 2018

@author: wfok007
"""
import matplotlib
matplotlib.use('agg')


import pickle
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

import time, math
import os
from decompress import addAxis, fast_transfer

from weightNorm import createModel, Dice_Loss, non_differentiable_Dice_coef, initialise_averager, load_stat, prepareInputsForGPU
from NNfunctions import metaGen


def train(gaa_mean, gaa_std, num_ranks, hpc, model, lrList, fupdate, b, strata_mean, strata_std):
    """
    This function trains a neural network with progressive change in the learning rate.
    
    Inputs:
        gaa_mean, gaa_std = input statistics
        num_ranks, hpc = information used in setting up training dataset (a generator)
        model = neural network
        lrList = a list of learning rates to be tested
        fupdate = how many updates we spent at each value of the learning rate
        b = minibatch size
        strata_mean, strata_std = a list of past performance statistics / Dice coefficients recorded
    
    Outputs:
        appending the performance statistics on strata_mean, strata_std 
        
    """

    minibatch = b

    generator_training = metaGen('train', num_ranks, hpc)
     
    curve_mean = {lr: [] for lr in lrList}
    curve_std = {lr: [] for lr in lrList}
    
    start_time = time.time()
    # loop through all learning rates
    for learning_rate in lrList:
        print ('current learning rate %g / %f' %(learning_rate, np.log10(learning_rate)))
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=learning_rate)
        grad_averager = initialise_averager(model)
        
        
        for ni in range(fupdate):
            print ('%d out of %d rounds' %(ni, fupdate))
            holder_coeff = []
            for i in range(minibatch):
                print ('%d out of %d minibatch' %(i, minibatch))
                # get a new sample from dataset
                full_image, full_mask, image, mask  = next(generator_training)
                image1 , target = prepareInputsForGPU(image, full_mask, gaa_mean, gaa_std)
                # zero the parameter gradients
                optimizer.zero_grad()
                
                output = model(image1)
                coef, true_pos_arr, false_pos_arr, false_neg_arr = accuracy(output, target, returnArr=True)
                loss = criterion(output, target)
                del target, output # cut memory usage
                
                loss.backward()
                
                # grab and calculate running averages on the gradient
                for name, parameter in model.named_parameters():
                    # because of pre-training, get trainable parameters
                    if parameter.requires_grad:
                       
                        current_grad = parameter.grad.clone()
                        
                        grad_averager[name].update(current_grad)
                

                holder_coeff.append(coef) # recording Dice coefficients

                del loss, image1

            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.grad = grad_averager[name].past
                    


            optimizer.step()
            grad_averager = initialise_averager(model)
            
            coMean = np.mean(holder_coeff)
            coStd = np.std(holder_coeff)
            
            print ('learning rate [%f] \t dice coefficient mean %f std %f' %(learning_rate, coMean, coStd))
            print ('testing learning rate %f took %f minutes' %(learning_rate, (time.time()-start_time)/60))
            curve_mean[learning_rate].append(coMean)
            curve_std[learning_rate].append(coStd)
            
    strata_mean[b].append(curve_mean)
    strata_std[b].append(curve_std)
    return strata_mean, strata_std
                    
def frange(start, stop, step):
    ''' Python implementation that is similar to np.arange()
    
    Inputs:
        start = initial value
        stop = terminal value
        step = gap size
    
    Output:
        generator, one vlaue at a time
    
    '''
    i = start
    while i < stop:
        yield i 
        i += step                 
                
if __name__ == '__main__':
    
    plt.close('all') # close all figure windows
    
    prog_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training\training'
    training_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training'
    output_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training\transformation\ResNet'
    debug_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training\debug'
    hpc = r'Z:\mpi_heart'
    weight_dir = 'Z:\mpi_heart\weight_checkpoints'

    
    mean, std = load_stat()
    
    num_ranks = 375
    od, oh, ow = 88 , 112, 112
    multiple = 0.25
    
    
    aa_mean, aa_std = addAxis(mean, std)
    gaa_mean, gaa_std = fast_transfer(aa_mean), fast_transfer(aa_std)

    criterion = Dice_Loss()
    accuracy = non_differentiable_Dice_coef()
    
    batchList = [5]
    # must sort the learning rates from tiny to huge. Otherwise we have the effect of learning rate decay
    # use log10
    lrList = sorted([1*math.pow(10, -i) for i in frange(0,4.2,0.5)])
    for lr in lrList:
        print ('%g: %f' %(np.log10(lr), lr))

    fupdate = 10 # spend 10 updates at each rate

    
    strata_mean = {b: [] for b in batchList} # results holder
    strata_std = {b: [] for b in batchList}
    for b in batchList:
        print ('current batch size is %d' %b)
        # create the neural network outside of the training loop
        # so that we can keep training it with different learning rates
        model = createModel(od, oh,ow, multiple) 
        strata_mean, strata_std = train(gaa_mean, gaa_std, num_ranks, hpc, model, lrList, fupdate, b, strata_mean, strata_std)
        
        del model # save memory
        
        # show the intermediate results
        plt.figure()
        y = [np.mean(strata_mean[b][0][lr]) for lr in lrList]
        yerr =  [np.mean(strata_std[b][0][lr]) for lr in lrList]
        plt.errorbar(np.log10(lrList), y, yerr=yerr)
        plt.title("batch size %d" %b)
        plt.ylim(0, 1)
        plt.grid()
        plt.savefig(os.path.join(prog_dir, str(b)+'_LRtest.png'))
        plt.close()
        

    print('Finished Learning Rate Test')
    pickle.dump( strata_mean, open( os.path.join(prog_dir, "strata_mean.p"), "wb" ) )
    pickle.dump( strata_std, open( os.path.join(prog_dir, "strata_std.p"), "wb" ) )

