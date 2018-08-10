# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 18:46:28 2018

@author: wfok007
"""

import numpy as np

import matplotlib.pyplot as plt
import os, itertools

if __name__ == '__main__':
    
    plt.close('all')
    
    prog_dir = r'Z:\mpi_heart\ensembleTestPredictions'
    training_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training'
    output_dir = r'Z:\mpi_heart\howBigEnsemble'
    debug_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training\debug'
    hpc = r'Z:\mpi_heart'
    
    coef_holder = np.load(os.path.join(prog_dir, 'coef_holder2.npy'))
    median = np.median(coef_holder, axis=1)
    ind = np.argsort(median) # lowest to the highest Dice coefficients
    # select only 1 in 4, otherwise the plot will be too crowded
    ind = ind[0:len(median):4]

    
    minibatch = 5
    
    weightNameList = [1480, 2440, 2880, 3320, 3920, 4440]
    weightNameList = ['weights'+str(i*minibatch) for i in weightNameList]
    
    # a dictionary that maps the name of the checkpoint to the number of columns
    weight_col_dict = {w:i for i, w in enumerate(weightNameList)}
    
    
    # combinations of different sizes of the ensemble (network using weights at different checkpoints)
    cpCombinations = {i:list(itertools.combinations(weightNameList,i)) for i in range(1,7)}
    
    total_num_combinations = sum([len(cpCombinations[key]) for key in sorted(cpCombinations.keys())])
    weightHeatMap = np.zeros((total_num_combinations, len(weightNameList)))
    
    counter = 0
    for key in sorted(cpCombinations.keys()):
        print (key)
        currentCombination = cpCombinations[key]
        print ('current combinations {}'.format(currentCombination))
        for groupMembers in currentCombination:
            for member in groupMembers:
                weightHeatMap[counter, weight_col_dict[member]] = 1
            counter = counter + 1
    
    fig, ax = plt.subplots()
    # use black and white for brevity
    im = ax.imshow(weightHeatMap[ind,:], cmap='Greys')
    ax.set_xticks(np.arange(len(weightNameList)))
    ax.set_yticks(np.arange(len(ind)))
    
    weightNameList = [1480, 2440, 2880, 3320, 3920, 4440]
    checkpoints = [str(w)for w in weightNameList]
    ax.set_xticklabels(checkpoints)
    
    results = [str(np.round(median[i], decimals=3)) for i in ind]
    ax.set_yticklabels(results)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    #ax.set_title("The median of Dice coefficients \n produced by different ensemble of neural networks")
    fig.tight_layout()
    plt.ylabel('median of Dice coefficients')
    plt.xlabel('checkpoints')
    plt.show()
            
    