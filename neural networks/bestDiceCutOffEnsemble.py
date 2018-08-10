# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 17:28:26 2018

@author: wfok007
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from checkPredictions import numpy_dice_coefficient

if __name__ == '__main__':
    
    plt.close('all')
    prog_dir =r'Z:\mpi_heart\howBigEnsemble' 
    
    npzfile = np.load(os.path.join(prog_dir, '4_valid_ensemble_loss_predict_mask'+'.npz'))
    pred_mask= npzfile['pred_mask']
    ground_truth= npzfile['ground_truth']
    
    # create thresholds: discretized 0-1 with a step size of 0.01
    thresholds = np.arange(0,1,0.01)
    # calculate the coefficients at each cutoff
    coefs = [numpy_dice_coefficient(pred_mask, ground_truth, threshold) for threshold in thresholds]
    
    # plot and see
    plt.figure()
    plt.plot(thresholds, coefs, label='ensemble of 4')
    plt.xlabel('thresholds')
    plt.ylabel('dice coefficient')
    plt.ylim([0,1])
    plt.legend()
    plt.grid()
    plt.show()
    
    # find the best cutoff
    print ('best threshold %f' %thresholds[coefs.index(max(coefs))])