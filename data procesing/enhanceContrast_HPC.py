# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 18:29:17 2018

@author: wfok007
"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os, copy
from PIL import Image
from readImages_HPC import getFilePaths
from matplotlib.colors import Normalize
from skimage import exposure

def plot_image_and_hist(image, mask, mask_cmap, img_cmap, axes, bins=256):
    """
    Plot MR images plus mask and their histogram and cumulative histogram.

    """
    flippedMask = -1*mask + 1
    
    # add transparency
    alphas = np.ones(image.shape)
    alphas = alphas * flippedMask
    alphas = np.clip(alphas, 0.7,1)

    image = image - image.min()
    image = image / image.max()

    colors = Normalize(0, 1, clip=True)(image)
    colors = img_cmap(colors)

    # Now set the alpha channel to the one we created above
    colors[..., -1] = alphas
    

    ax_image, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_image.imshow(mask, cmap=mask_cmap)
    ax_image.imshow(colors, cmap=img_cmap)
    ax_image.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    image_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, image_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_image, ax_hist, ax_cdf

def adaptiveContrast(image, mask, target_path, name, kernel_sizes, save=False):
    """
    use adaptive contrast equalization to adjust the intensity
    
    inputs:
        image = MR (single slice)
        mask = image mask
        target_path = file directory
        name = the ID of the current slice
        kernel_sizes = a list of the dimension of the kernels in pixels
        save = whether to save the file to disk for debugging only
    
    output:
        image_adapteq = numpy array of the adjusted image
    
    """

    transforms = []
    for kernel_size in kernel_sizes:
        image_adapteq = exposure.equalize_adapthist(image, kernel_size=kernel_size, clip_limit=0.03)
        transforms.append(image_adapteq)
    
    # Display results
    fig = plt.figure(figsize=(19, 16))
    axes = np.zeros((2, 5), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 5, 1)
    for i in range(1, 5):
        axes[0, i] = fig.add_subplot(2, 5, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, 5):
        axes[1, i] = fig.add_subplot(2, 5, 6+i)
    
    ax_image, ax_hist, ax_cdf = plot_image_and_hist(transforms[0], mask, mask_cmap, img_cmap,
                                                    axes[:, 0])
    ax_image.set_title('%d' %kernel_sizes[0])
    
    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))
    
    ax_image, ax_hist, ax_cdf = plot_image_and_hist(transforms[1], mask, mask_cmap, img_cmap,
                                                    axes[:, 1])
    ax_image.set_title('%d' %kernel_sizes[1])
    
    ax_image, ax_hist, ax_cdf = plot_image_and_hist(transforms[2], mask, mask_cmap, img_cmap,
                                                    axes[:, 2])
    ax_image.set_title('%d' %kernel_sizes[2])
    
    ax_image, ax_hist, ax_cdf = plot_image_and_hist(transforms[3],mask, mask_cmap, img_cmap,
                                                    axes[:, 3])
    ax_image.set_title('%d' %kernel_sizes[3])
    
        
    ax_image, ax_hist, ax_cdf = plot_image_and_hist(transforms[4],mask, mask_cmap, img_cmap,
                                                    axes[:, 4])
    ax_image.set_title('%d' %kernel_sizes[4])
    
    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))
    
    # prevent overlap of y-axis labels
    fig.tight_layout()
    if save:
        plt.savefig(os.path.join(target_path, name))
    else:
        plt.show()
    plt.close()

    return image_adapteq
    

def enhanceContrast(image, mask, target_path, name, save=False):
    
    """
    plotting the different effects of different algorithms (none, contrast
    stretching, histogram equalization, adaptive equalization)
    """
    

    
    # Contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    image_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    
    # Equalization
    image_eq = exposure.equalize_hist(image)
    
    # Adaptive Equalization
    image_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
    
    # Display results
    fig = plt.figure(figsize=(19, 13))
    axes = np.zeros((2, 4), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5+i)
    
    ax_image, ax_hist, ax_cdf = plot_image_and_hist(image, mask, mask_cmap, img_cmap,
                                                    axes[:, 0])
    ax_image.set_title('Low contrast image')
    
    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))
    
    ax_image, ax_hist, ax_cdf = plot_image_and_hist(image_rescale, mask, mask_cmap, img_cmap,
                                                    axes[:, 1])
    ax_image.set_title('Contrast stretching')
    
    ax_image, ax_hist, ax_cdf = plot_image_and_hist(image_eq, mask, mask_cmap, img_cmap,
                                                    axes[:, 2])
    ax_image.set_title('Histogram equalization')
    
    ax_image, ax_hist, ax_cdf = plot_image_and_hist(image_adapteq,mask, mask_cmap, img_cmap,
                                                    axes[:, 3])
    ax_image.set_title('Adaptive equalization')
    
    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))
    
    # prevent overlap of y-axis labels
    fig.tight_layout()
    if save:
        plt.savefig(os.path.join(target_path, name))
    else:
        plt.show()
    plt.close()
    return image_adapteq

def handle(im, isImage):
    '''
    this function modifies file type.
    
    Inputs:
    im: image in PIL format
    isImage: boolean to show if the input is mask or an image
    
    Outputs:
    im: image or mask in numpy array format    
    
    '''
    
    if isImage:
        # keep only the first 3 channels, discard alpha channel
        # format
        im = np.array(im)
        #print (im.shape)
        im = im[:,:,0:-1]
        im = im[np.newaxis,...]
        im = np.swapaxes(im, -1, 0)
        im = np.squeeze(im)
        return im
    else:
        im = np.array(im)
        # check which channel is useful
        # because mask has got only 1 useful channel
        dim = im.shape[-1]
        useful = None
        for d in range(dim):
            if im[0,0,d] != 255:
                # that is useful
                useful = d
        if useful is not None:
            return im[:,:,useful]
    
def resize(array, size, cmap, isImage):
    '''
    this function helps resize images or mask.
    
    Inputs:
    array: numpy array of one single image or its mask
    size: tuple of the new sizes
    cmap : colormap for image or mask
    isImage: boolean variable to distinguish an image or mask
    
    Outputs:
    im_pil: the PIL format
    array: numpy format    
    '''
    if array.dtype != np.uint8:
        im = Image.fromarray(np.uint8(cmap(array)*255))
    else:
        im = Image.fromarray(array)
    im = im.resize(size, resample=Image.BICUBIC)
    im_pil = copy.copy(im)
    return handle(im, isImage), im_pil

def enhance(images2, masks3):
    
    '''
    This function applies adaptive contrast equalization to these images.
    It repeatedly applies a series of kernel sizes / filters on these images.
    User can decide how many scales or filters are used. Currently it is 2.
    
    Inputs:
    images2 : stack of images
    masks3 = stack of their masks
    
    Outputs:
    img_stacks : contrast-adjusted image stack
    mask_stacks : their masks
    
    '''
    
    print ('image input shape {}'.format(images2.shape))
    num_samples, depth, old_height, old_width = images2.shape
    channels = 3 # RGB 
    # use resize as a reformatting function
    height = old_height
    width = old_width
    
    # grid search suitable kernel size
#    kernel_fractions = [1/4, 1/16, 1/32, 1/64, 1/128]
    kernel_fractions = [1/8, 1/32]
    
    
    kernel_sizes = [int(old_height*i) for i in kernel_fractions]
    print ('applying kernel sizes of {}'.format(kernel_sizes))
    
    num_kernels = len(kernel_fractions)
    
    # pre-allocate memory
    img_stacks = np.zeros((num_samples, depth*num_kernels, channels, old_height, old_width), dtype=np.uint8)
    mask_stacks = np.zeros((num_samples, depth, old_height, old_width), dtype=np.uint8) 
  
    for n in range(num_samples):
        print ('working on sample %d' %n)
                
        mask_list = []
        for k in range(num_kernels):
            img_list = []
            print ('kernel %d' %kernel_sizes[k])
            for counter in range(depth):
                patient_code = os.path.basename(os.path.dirname(image_paths[n]))
                # use the patient id
                #name = patient_code + "_"+str(counter)
                target_path = os.path.join(output_dir,patient_code)
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                
                image_adapteq = exposure.equalize_adapthist(images2[n,counter,:,:],
                                                            kernel_size=kernel_sizes[k], clip_limit=0.03)

#                # reformatting
                img, img_pil = resize(image_adapteq, (height, width), img_cmap, isImage=True)
                img_stacks[n, counter+k*depth, ...] = img
                img_list.append(img_pil)
                
                # only process the mask once
                if k == 0:
                    mask = masks3[n, counter,:,:] * 1
                    mask, _ = resize(mask, (height,width), mask_cmap, isImage=False)
                    mask_stacks[n, counter,...] = mask          
                    mask_list.append(mask)

    return img_stacks, mask_stacks

if __name__ == '__main__':
    
    # my default colormaps for mask and images
    mask_cmap = plt.cm.spring
    img_cmap = plt.cm.gray
    
    mask_paths, image_paths = getFilePaths()

    training_dir = os.getcwd()
    output_dir = r'/hpc/wfok007/mpi_heart/pairs'
    
    npzfile = np.load(os.path.join(training_dir, 'split2.npz'))


    images2, masks3  = npzfile['train_images'], npzfile['train_targets']
    train_images, train_targets = enhance(images2, masks3)
    
    images2, masks3  = npzfile['valid_images'], npzfile['valid_targets']
    valid_images, valid_targets = enhance(images2, masks3)
    
    images2, masks3  = npzfile['test_images'], npzfile['test_targets']
    test_images, test_targets = enhance(images2, masks3)
    
    
    
    np.savez_compressed(os.path.join(training_dir, 'pairs_input_data2'),
                        train_images=train_images,
                        valid_images=valid_images,
                        test_images=test_images,
                        train_targets=train_targets,
                        valid_targets=valid_targets,
                        test_targets=test_targets)