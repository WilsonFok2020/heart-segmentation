# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:29:50 2018

@author: wfok007
"""
#from __future__ import print_function, division
#import matplotlib
#matplotlib.use('agg')

import torch, math
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt

import time
import os
from transform import  save, fromFloat_to_uint8
from runningAvg import averager
from decompress import decompress_mask, addAxis, fast_transfer, gpuDecompress, gpuDecompressMask, gpuNormalise
from NNfunctions import interpolate3D, metaGen, freeze_layer

def createCyclical(T, learning_rate, version, min_fraction=1/3):
    
    ''' 
    This function creates a lookup table. The rows are values of sin(t), and the 
    columns are values of a linear learning rate function
    
    Inputs:
        T = is the period of the cycle
        learning rate = max rate
        version = high to low or low to high
        min_fraction = a fraction that calculates the slowest allowable learning rate
        
    Outputs:
        lookup = dictionary / the table
        storedKeys = dictionary key
        lrs = values of rates (in fraction) in a single complete cycle
    '''
    
    # no endpoint as we cycle all items anyway
    
    if version =='highLow':
        
        lrs = [np.linspace(1,min_fraction,int(T/2),
                           endpoint=False),
                np.linspace(min_fraction,1,int(T/2), endpoint=False)]
    elif version =='lowHigh':
        lrs = [np.linspace(min_fraction,1,int(T/2),
                           endpoint=False),
                np.linspace(1,min_fraction,int(T/2), endpoint=False)]
        
    lrs = np.round(np.hstack(lrs), decimals=2).tolist()
    print ('maximum learning rate {}'.format(learning_rate*lrs[0]))
    print ('minium learning rate {}'.format(learning_rate*lrs[int(T/2)]))
    print ('number of f updates or bites %d' %len(lrs))
    
    # mapping sin value to index to imporve on the flat part of the sin curve
    
    lookup = dict()
    for i in range(T):
        # as many decimals places as possible to get unique keys
        s = np.sin(np.pi*(i/T))
       
        lookup[s] = i
        
    storedKeys = list(lookup.keys())
    return lookup, storedKeys, lrs
    
def triangular(i):
    """
    this function helps to look up the right fraction to apply to
    the learning rate at any point in the cycle. Sine is handy for it is
    an even function.
    
    Inputs:
        i = counter created by pytorch
    output:
        the fraction at which the i point in time in a cycle.
    
    """
    
    s = np.abs(np.sin(np.pi*(i/T)))
    difference = [abs(k-s) for k in storedKeys]
    keyS =  storedKeys[difference.index(min(difference))]
    index = lookup[keyS]
    return lrs[index]


class Net(nn.Module):
    '''
    The main part of the network
    
    Inputs:
        original_model = pre-train model
        od = input depth
        oh =input height
        ow = input width
        multiple = fraction that adjusts the number of filters in the network
        kernels = number of stacks of inputs after intensity equalization with various kernels' sizes
        
    
    more pre-trained resnet can be found
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    '''
    def __init__(self, original_model, od, oh, ow, multiple, kernels=2):
        super().__init__()
        # get the architecture of the pre-trained network
        children = list(original_model.children())
        
        self.kernels = kernels
        self.conv1 = nn.Sequential(*children[0:3]) # skip pooling
        self.conv2 = nn.Sequential(*children[4])
        self.conv3 = nn.Sequential(*children[5])
        self.conv4 = nn.Sequential(*children[6])
        self.conv5 = nn.Sequential(*children[7])
        
        self.d5 = nn.Sequential(nn.Conv3d(int(512*multiple), int(256*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(256*multiple)),
                                  nn.ReLU())
        
        self.h4 = nn.Sequential(nn.Conv3d(int(512*multiple),int(256*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(256*multiple)),
                                  nn.ReLU())
        
        self.d4 = nn.Sequential(nn.Conv3d(int(256*multiple), int(128*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(128*multiple)),
                                  nn.ReLU())
        
        self.h3 = nn.Sequential(nn.Conv3d(int(256*multiple), int(128*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(128*multiple)),
                                  nn.ReLU())
                
        self.d3 = nn.Sequential(nn.Conv3d(int(128*multiple), int(64*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(64*multiple)),
                                  nn.ReLU())
        
        self.h2 = nn.Sequential(nn.Conv3d(int(128*multiple),int(64*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(64*multiple)),
                                  nn.ReLU())
        
        self.d2 = nn.Sequential(nn.Conv3d(int(64*multiple), int(32*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(32*multiple)),
                                  nn.ReLU())

        self.s5 = nn.Sequential(nn.Conv3d(int(512*kernels), int(512*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(512*multiple)),
                                  nn.ReLU())
        self.s4 = nn.Sequential(nn.Conv3d(int(256*kernels), int(256*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(256*multiple)),
                                  nn.ReLU())
        self.s3 = nn.Sequential(nn.Conv3d(int(128*kernels), int(128*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(128*multiple)),
                                  nn.ReLU())
        self.s2 = nn.Sequential(nn.Conv3d(int(64*kernels), int(64*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(64*multiple)),
                                  nn.ReLU())
        self.s1 = nn.Sequential(nn.Conv3d(int(64*kernels), int(64*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(64*multiple)),
                                  nn.ReLU())
        self.s0 = nn.Sequential(nn.Conv3d(int(64*multiple), int(64*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(64*multiple)),
                                  nn.ReLU())
        
        self.h0 = nn.Sequential(nn.Conv3d(int(128*multiple), int(64*multiple), kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False),
                                  nn.BatchNorm3d(int(64*multiple)),
                                  nn.ReLU())


        

        self.nin = nn.Conv3d(int(32*multiple)+2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        
        self.inter1 = interpolate3D(88,14,14)
        self.inter2 = interpolate3D(88,28,28)
        self.inter3 = interpolate3D(88,56,56)
        self.inter4 = interpolate3D(od,oh,ow)
        self.inter5 = interpolate3D(88, 576, 576)
        
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        
        x = x[:,0,:,:] # get rid of the 3 channels
        x = x.unsqueeze(0)
        x = x.view(x.shape[0], self.kernels, 88, x.shape[2], x.shape[3])

        xs = [x1, x2, x3, x4, x5]
        xs = [i.unsqueeze(0) for i in xs]
        xs = [i.view(self.kernels, 88, i.shape[2], i.shape[3], i.shape[4]) for i in xs]
        
        
        xs = [i.permute(0,2,1,3,4) for i in xs]
        x1, x2, x3, x4, x5 = xs

        
        # fragmented footprint, need to bring dimensions together first
        xs = [i.contiguous() for i in xs]
        xs = [i.view(1, -1, i.shape[2], i.shape[3], i.shape[4]) for i in xs]
        x1, x2, x3, x4, x5 = xs
        
        print ('x1 shape{}'.format(x1.shape))
        print ('x2 shape{}'.format(x2.shape))
        print ('x3 shape{}'.format(x3.shape))
        print ('x4 shape{}'.format(x4.shape))
        print ('x5 shape{}'.format(x5.shape))
        

#        
        x15 = self.s5(x5)
        x14 = self.s4(x4)
        x13 = self.s3(x3)
        x12 = self.s2(x2)
        x11 = self.s1(x1)
        
        print ('x15 shape{}'.format(x15.shape))
        print ('x14 shape{}'.format(x14.shape))
        print ('x13 shape{}'.format(x13.shape))
        print ('x12 shape{}'.format(x12.shape))
        print ('x11 shape{}'.format(x11.shape))
        
        x25 = self.d5(x15)

        x25 = self.inter1(x25)
        
        x24 = self.h4(torch.cat((x14, x25), 1))


        x34 = self.d4(x24)
        
        x34 = self.inter2(x34)
        
        x23 = self.h3(torch.cat((x13, x34), 1))
        x33 = self.d3(x23)
        x33 = self.inter3(x33)
        
        print ('x25 shape{}'.format(x25.shape))
        print ('x24 shape{}'.format(x24.shape))
        print ('x34 shape{}'.format(x34.shape))
        print ('x23 shape{}'.format(x23.shape))
        print ('x33 shape{}'.format(x33.shape))

        x22 = self.h0(torch.cat((x12, x33), 1))
        x32 = self.s0(x22)
        x21 = self.h2(torch.cat((x11, x32), 1))
        x21 = self.d2(x21)
        
        
        x0 = self.inter4(x21)
        x00 = torch.cat((x0, x), 1)
        seg = self.inter5(self.nin(x00))
        

        
        print ('x22 shape{}'.format(x22.shape))
        print ('x32 shape{}'.format(x32.shape))
        print ('x21 shape{}'.format(x21.shape))
        print ('x0 shape{}'.format(x0.shape))
        print ('x00 shape{}'.format(x00.shape))
        
        
        return seg

def flip(arr):
    """
    toggle between 0 and 1
    """
    
    return arr*(-1) + 1

def torch_flip(arr):
    '''
    a = torch.tensor([[1, 0], [0, 1],[1,1], [0,0]])
    
    this line is wrong, torch sum is a reduced sum not elementwise sum
    return torch.sum(torch.mul(arr, -1), 1, keepdim=True)
    
    cannot do this as well
    c = torch.sum(b, torch.ones(4,2))
    '''
    return torch.mul(arr, -1) + 1
#    
def cal_Dice_Similarity_Coefficient(pred_mask, mask):
    
    ''' 
    This function uses numpy to calculate Dice coefficient
    
    inputs: 
        pred_mask int 32 0 or 1 (depth, height, width)
        mask/ ground truth float 32 0 or 255 (same dimension) '''
    
    print ('numpy dice ')
    mask = mask/255
    
    true_pos = np.sum(np.multiply(pred_mask, mask))
    true_pos = true_pos / 88
    flipped_mask = flip(mask)
    false_pos = np.sum(np.multiply(flipped_mask, pred_mask))
    false_pos = false_pos / 88
    flipped_pred = flip(pred_mask)
    false_neg = np.sum(np.multiply(mask, flipped_pred))
    false_neg = false_neg/ 88
    n = 2*true_pos
    dice = n/(n+false_neg+false_pos)
    print ('true pos %f'%true_pos)
    print ('false pos %f' %false_pos)
    print ('false_neg %f' %false_neg)
    print ('n %f'%n)
    print ('Dice similarity coefficient %f' %dice)
    return dice

class non_differentiable_Dice_coef(torch.nn.Module):
    
    """
    This function uses pytorch to calculate Dice coefficient

    """
    def __init__(self):
        super().__init__()
    def forward(self, output, target, threshold=0.5, returnArr=True):

        pred_mask = nn.functional.sigmoid(output)
        # no need to store the gradient because this is not used in the objective function
        pred_mask = torch.autograd.Variable(pred_mask.gt(threshold),requires_grad=False)
        # make sure the variable is of the right type
        pred_mask = pred_mask.type(torch.cuda.FloatTensor)

        
        true_pos_arr = torch.mul(pred_mask, target)
        true_pos = torch.sum(true_pos_arr)
        true_pos = true_pos / 88
        flipped_mask = torch_flip(target)
        
        false_pos_arr = torch.mul(flipped_mask, pred_mask)
        false_pos = torch.sum(false_pos_arr)
        false_pos = false_pos / 88
        flipped_pred = torch_flip(pred_mask)
        
        false_neg_arr = torch.mul(target, flipped_pred)
        false_neg = torch.sum(false_neg_arr)
        false_neg = false_neg/ 88
        
        n = torch.mul(true_pos,2)
        dice = torch.div(n, (n+false_neg+false_pos))
        
        if returnArr:
            # get the item, otherwise we have a pytorch tensor (scalar)
            return dice.item(), true_pos_arr.cpu().detach().numpy(),\
                false_pos_arr.cpu().detach().numpy(), false_neg_arr.cpu().detach().numpy()
        else:
            return dice.item()

class Dice_Loss(torch.nn.Module):
    """
    This function calculates Dice loss which is used in the objective function.
    """
    def __init__(self):
        super().__init__()
    def forward(self, output, target, smooth=1):
        pred_mask = nn.functional.sigmoid(output)
        a = torch.mul(pred_mask, target)

        depth = a.shape[2]
        b = a.view(depth, -1)
        
        c = torch.sum(b, dim=1)
        
        c = 2*c
        # just in case there is not 1 in the prediction or target
        c = c + smooth
        
        pred_square = torch.mul(pred_mask, pred_mask)
        target_square = torch.mul(target, target)
        
        pred_square = pred_square.view(depth, -1)
        target_square = target_square.view(depth, -1)
        ps = torch.sum(pred_square, dim=1)
        ts = torch.sum(target_square, dim=1)
        
        denominator = ps+ts
        denominator = denominator + smooth
        # minimize, so 1-
        slice_by_slice_loss = 1- torch.div(c, denominator)
        return torch.sum(slice_by_slice_loss)
        
def initialise_averager(model):
    """ 
    this function initializes a average for every trainable parameters
    This average takes the mean of a minibatch.
    Inputs:
        model = network
    
    output:
        dictionary of parameters and their averages
    """

    grad_averager = {}
    for name, parameter in model.named_parameters():
        # because of pre-training, get trainable parameters
        if parameter.requires_grad:
            grad_averager[name] = averager()
    return grad_averager
          


def validation(i, gaa_mean, gaa_std,generator_valid, model, criterion, accuracy, prog_dir):
    """
    This function validates the network whenever it is called. It randomly selects
    one sample in the validation dataset to valid the network.
    It is possible to valid more than one sample but in the interest of time, only one 
    sample is used for validation.
    
    The current step is i
    
    gaa_mean, gaa_std = average intensity and standard deviation used for normalization
    
    generator_valid = a generator that holds the validation dataset
    model = network
    criterion = Dice loss
    accuracy = Dice coefficient
    prog_dir = storage directory
    
    Outputs:
        loss = Dice loss
        coef = Dice coefficient
    
    """
    
    print ('--------------------validation------------------')
    full_image, full_mask, image, mask  = next(generator_valid)
    image1 , target = prepareInputsForGPU(image, full_mask, gaa_mean, gaa_std)
        
    # no need to back-propagate
    with torch.no_grad():
        output = model(image1)
        loss = criterion(output, target)
        coef, true_pos_arr, false_pos_arr, false_neg_arr = accuracy(output, target, returnArr=True)
        sig = nn.functional.sigmoid(output)
        sig = sig.cpu().detach().numpy()
        pred_mask = sig[0,0,...]
        
    del output, sig, image1, target
    val_loss = loss.item()
    print ('validation loss %f; dice coef: %f' %(val_loss, coef))

    # use a default cutoff of 0.5
    pred_mask = pred_mask > 0.5
    pred_mask = pred_mask *1
    # save validation predictions and mistakes for visualization/ inspection
    temp_img = fromFloat_to_uint8(full_image)
    
    save(temp_img, pred_mask, 'validation_'+ str(i), prog_dir)
    
    # need to decompress the index to get mask
    mask = decompress_mask(full_mask,88, 576, 576)
    save(temp_img, mask, 'validation_groundTruth_'+str(i), prog_dir)
#    
    save(temp_img, true_pos_arr[0,0,...], 'validation_true_pos_'+str(i), prog_dir)
    save(temp_img, false_pos_arr[0,0,...], 'validation_false_pos_'+str(i), prog_dir)
    save(temp_img, false_neg_arr[0,0,...], 'validation_false_neg_'+str(i), prog_dir)
    
    return loss.item(), coef

def plot_norm(i, normHist, holder, prog_dir):
    """
    This function plots the changing norm of the error derivatives
    Inputs:
        i = current step
        normHist = past norm magnitude
        holder = the step at which the norm was recorded
        prog_dir = save to this directory
    """
    
    names = list(normHist.keys())
    plt.figure()

    for name in names:
        if name[0] == 'd':
            plt.plot(adjustForMinibatch(holder), normHist[name], label=name)
    plt.xlabel('number of times weights being updated')
    plt.legend()
    plt.savefig(os.path.join(prog_dir, 'normD_'+str(i)+'.png'))
    plt.close()
    
    plt.figure()
    for name in names:
        if name[0] == 's':
            plt.plot(adjustForMinibatch(holder), normHist[name], label=name)
    plt.xlabel('number of times weights being updated')
    plt.legend()
    plt.savefig(os.path.join(prog_dir, 'normS_'+str(i)+'.png'))
    plt.close()
    
    plt.figure()

    for name in names:
        if name[0] != 's' and name[0] != 'd':
            plt.plot(adjustForMinibatch(holder), normHist[name], label=name)
    plt.xlabel('number of times weights being updated')
    plt.legend()
    plt.savefig(os.path.join(prog_dir, 'norm_'+str(i)+'.png'))
    plt.close()
    

def addCheckpoints(checkpoint_hist):
    """
    this function adds a vertical line on a figure at locations specified in 
    checkpoint_hist
    """
    if len(checkpoint_hist) >0:
        for xc in adjustForMinibatch(checkpoint_hist):
            plt.axvline(x=xc, linestyle='--')
    

def adjustForMinibatch(trainHolder, minibatch=5):
    """
    As the step number is based on the number of times a case is fed to the network,
    not the number of times network updates itself, this function converts
    "feeding" to "updates"
    
    """
    return [t/ minibatch for t in trainHolder]
def plot_loss(i, trainHolder, trainLoss, validHolder, validLoss, name, checkpoint_hist, prog_dir):
    """
    This function puts the Dice loss over past weight updates on a chart.
    
    i = current step
    trainHolder = step at which loss was recorded during training
    trainLoss = the recorded loss
    validHolder = step at which loss was recorded during validation
    validLoss = recording 
    name = output filename
    checkpoint_hist = steps at which weights/ checkpoints were taken
    prog_dir = storage directory
    
    """
    
    plt.figure()
    plt.scatter(adjustForMinibatch(trainHolder), trainLoss, label='train') 
    plt.scatter(adjustForMinibatch(validHolder), validLoss, label='validation') 
    addCheckpoints(checkpoint_hist)
    plt.xlabel('number of times weights being updated')
    plt.ylabel(name)
    plt.legend()
    plt.savefig(os.path.join(prog_dir, name+str(i)+'.png'))
    plt.close() 

def plot_learningRate(i, normHolder, checkpoint_hist, lr_history):
    """
    This function puts the learning rates during.
    
    i = current step
    normHolder = step at which rate was recorded during training
    checkpoint_hist = steps at which weights/ checkpoints were taken
    lr_history = past rate values
    
    """
    
    plt.figure()
    plt.plot(adjustForMinibatch(normHolder), lr_history)
    addCheckpoints(checkpoint_hist)
    plt.xlabel('number of times weights being updated')
    plt.ylabel('learning rate')
    plt.savefig(os.path.join(prog_dir, 'learningRate'+str(i)+'.png'))
    plt.close() 

    

def init_weights(m):
    """
    apply weight initialization scheme to a model with a particular layer type
    """
    
    if type(m) == nn.Conv3d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def createModel(od, oh, ow, multiple):
    """
    This block creates a model, then initialize its weights and put it onto the GPU
    Inputs:
        od, oh, ow, multiple = network specifications
    Output
        model
    """
    
    original_model = torchvision.models.resnet34(pretrained=True)

    model = Net(original_model, od, oh, ow, multiple)
    model.apply(init_weights)
    
    # freeze the resNet feature map parts
    for layer in [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5]:
        freeze_layer(layer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)   
    model.to(device)
    
    return model

def load_stat(hpc=r'Z:\mpi_heart' , name ='kernel_overall_mean_std.npz'):
    """
    load the statistics of training data
    return the mean and standard derivation (numpy array)
    """
    
    npzfile = np.load(os.path.join(hpc, name))
    mean = npzfile['mean']
    std = npzfile['std']
    return mean, std

def prepareInputsForGPU(image, full_mask,gaa_mean, gaa_std):
    
    """
    This packs a series of operations that transfer input to GPU and then format and normalizes it.
    Inputs:
        image, full_mask = MR images and their masks
        gaa_mean, gaa_std = input statistics
    
    Outputs:
        image and target for neural network on GPU
    
    """
    
    # transfer input to GPU
    gImage = fast_transfer(image)
    gImage = gpuDecompress(gImage)
    
    gMask = fast_transfer(full_mask, isImage=False)
    gMask = gpuDecompressMask(gMask, 88, 576, 576)
    image1, target = gpuNormalise(gImage, gMask, gaa_mean, gaa_std)
    
    return image1, target    
if __name__ == '__main__':
    
    plt.close('all')
    
    prog_dir = r'Z:\mpi_heart\prog'
    training_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training'
    output_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training\transformation\ResNet'
    debug_dir = r'C:\Users\wfok007\femuring\imaging\AtriaSeg_2018_training\debug'
    hpc = r'Z:\mpi_heart'
    weight_dir = 'Z:\mpi_heart\weight_checkpoints'

    
    mean, std = load_stat()
    
    num_ranks = 125
    num_ranks_train = 375
    od, oh, ow = 88 , 112, 112
    multiple = 0.25
    model = createModel(od, oh,ow, multiple)
    
    # put the statistics on the GPU
    aa_mean, aa_std = addAxis(mean, std)
    gaa_mean, gaa_std = fast_transfer(aa_mean), fast_transfer(aa_std)
    
    # initialize the data for validation and training
    generator_valid = metaGen('valid', num_ranks, hpc)
    generator_train = metaGen('train', num_ranks_train, hpc)

    criterion = Dice_Loss()
    accuracy = non_differentiable_Dice_coef()
    

    learning_rate = math.pow(10, -1.5)
    T = 50
    lookup, storedKeys, lrs = createCyclical(T, learning_rate, 'highLow', min_fraction=1/10)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=learning_rate)
    # useful for cyclical learning rate
    scheduler = lr_scheduler.LambdaLR(optimizer, triangular)
    

    grad_averager = initialise_averager(model)

    running_loss = 0.0
    running_coef = 0.0
    printout_freq = 100
    MAX_EPOCH = 10
    minibatch = 5
    saving_freq = 200
    validation_freq = 300
    
    trainLoss = []
    trainCo = []
    
    validLoss = []
    validCo = []
    
    normHist = {}
    trainHolder = []
    validHolder = []
    normHolder = []
    lr_history = []
    checkpoint_hist = []
    

    # grab and calculate running averages on the gradient
    for name, parameter in model.named_parameters():
        # because of pre-training, get trainable parameters
        if parameter.requires_grad:
            normHist[name] = list()
            
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print ('total number of parameters %d' %pytorch_total_params)    

    # restore to the last checkpoint
    river = 0
    for i in range(river):
        if i % minibatch  == 0  and i >= minibatch:
            scheduler.step()
    print ('scheduler at the right step')
        
    start_time = time.time()        
    for i, (full_image, full_mask, image, mask) in enumerate(generator_train):
        #print ('maximum allocated %f GB' %(torch.cuda.max_memory_allocated()/math.pow(10,9)))
        # restore to checkpoint
        if i == river:
            print ('i at the right step of %d' %river)
        elif i > river:
            
            
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
    
            
            if i % minibatch  == 0  and i >= minibatch:
                # only count the step for the scheduler whenever an update was made
                scheduler.step()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        parameter.grad = grad_averager[name].past
                        normHist[name].append(torch.norm(parameter.grad).cpu().numpy().tolist())
                
                normHolder.append(i)
                optimizer.step()
                print ('[%d] update weights time elasped: %f' %(i, time.time()-start_time))
                lr_history.append(scheduler.get_lr()[0])
                grad_averager = initialise_averager(model)
                
            # print statistics
            running_loss += loss.item()
            running_coef += coef
            del loss
            
            if i % printout_freq == 0 and i >= printout_freq-1:    # print every 2000 mini-batches
                avgLoss = running_loss / printout_freq
                avgAcc = running_coef / printout_freq
                print('[%d] loss: %f ; dice coef: %f' %(i, avgLoss, avgAcc ))
                trainLoss.append(avgLoss)
                trainCo.append(avgAcc)
                trainHolder.append(i)
                running_loss = 0.0
                running_coef = 0.0
                

            if i % saving_freq ==0 and i >= saving_freq-1:
                torch.save(model.state_dict(), os.path.join(weight_dir, 'weights'+str(i)))
                checkpoint_hist.append(i)
            
            del image1
            
            if i% validation_freq == 0 and i >= validation_freq-1:
                los,co = validation(i,gaa_mean, gaa_std, generator_valid, model, criterion, accuracy,prog_dir)
                validCo.append(co)
                validLoss.append(los)
                validHolder.append(i)
                
                plot_loss(i, trainHolder, trainLoss,validHolder, validLoss, 'loss', checkpoint_hist, prog_dir)
                plot_loss(i, trainHolder, trainCo, validHolder, validCo,'dice_coeff', checkpoint_hist, prog_dir)
                
                plot_norm(i, normHist, normHolder, prog_dir)
                plot_learningRate(i, normHolder, checkpoint_hist, lr_history)
        
        print ('[%d] maximum allocated %f GB' %(i, torch.cuda.max_memory_allocated()/math.pow(10,9)))

        if i > MAX_EPOCH: # terminate training if steps exceed a maximum value
            break

    print('Finished Training')
    