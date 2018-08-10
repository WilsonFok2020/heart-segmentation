# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:41:30 2018

@author: wfok007
"""

import numpy as np

def flip(arr):
    return arr*(-1) + 1
def numpy_dice_coefficient(pred_mask, target, threshold):
    ''' 
    This function computes the Dice coefficient on the CPU at a 
    specific threshold
    The calculation automatically sums over all minibatches or axis 0
    
    Inputs:
        pred_mas = prediction 
        target = ground truth of the same dimension
        threshold = cutoff value
    
    Output:
        Dice coefficient
        
        
    
    '''
    pred_mask = pred_mask > threshold
    true_pos_arr = np.multiply(pred_mask, target)
    true_pos = np.sum(true_pos_arr)
    flipped_mask = flip(target)
    
    false_pos_arr = np.multiply(flipped_mask, pred_mask)
    false_pos = np.sum(false_pos_arr)
    flipped_pred = flip(pred_mask)
    
    false_neg_arr = np.multiply(target, flipped_pred)
    false_neg = np.sum(false_neg_arr)

    
    n = np.multiply(true_pos,2)
    dice = np.divide(n, (n+false_neg+false_pos))

    return dice

if __name__ == '__main__':
    pass
    
    