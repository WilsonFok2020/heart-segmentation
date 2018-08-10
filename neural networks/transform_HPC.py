# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:03:31 2018

@author: wfok007
"""

from PIL import Image
import math
from math import floor, ceil
import numpy as np
import os, itertools
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def zoom_padding(xFactor, yFactor, w_dash, h_dash, w, h):
    '''
    this function adds zero to fill in the blanks left from cropping.
    
    Inputs:
    xFactor: zoom level for the width
    yFacto: zoom level for the height
    w_dash : new width
    h_dash: new height
    w : original width
    h : original height
    
    Outputs:
    border: a box that crops an image or mask
    
    '''
    
    if xFactor < 1 and yFactor < 1:
        border = (floor((float(w_dash) / 2) - (float(w) / 2)),
                  floor((float(h_dash) / 2) - (float(h) / 2)),
                  floor((float(w_dash) / 2) + (float(w) / 2)),
                  floor((float(h_dash) / 2) + (float(h) / 2)))
        
    elif xFactor >= 1 and yFactor < 1:
        left_shift = random.randint(0, floor(w_dash - w))
        border = (left_shift,
                  floor((float(h_dash) / 2) - (float(h) / 2)),
                  left_shift+w,
                  floor((float(h_dash) / 2) + (float(h) / 2)))
    elif xFactor < 1 and yFactor >= 1:
        down_shift = random.randint(0, floor(h_dash - h))
        border = (floor((float(w_dash) / 2) - (float(w) / 2)),
                  down_shift,
                  floor((float(w_dash) / 2) + (float(w) / 2)),
                  down_shift+h)
    else:
        left_shift = random.randint(0, floor(w_dash - w))
        down_shift = random.randint(0, floor(h_dash - h))
        border = (left_shift,
                  down_shift,
                  left_shift+w,
                  down_shift+h)
    
    return border
        
def zoom(stack, mask_stack):
    '''
    This function zooms in or out of images
    
    Inputs:
    stack: a list of images
    mask_stack: a list of mask
    
    Outputs:
    newStack2: list of zoomed and cropped images
    newMask2: their masks
        
    '''
    
    
    xFactor, yFactor =np.random.rand(2) + 0.5 # 0.5 - 1.5 zoom range
 
    image = stack[0]
    w,h = image.size
    w_dash = int(w*xFactor)
    h_dash = int(h*yFactor)

    
    newStack = [image.resize((w_dash, h_dash), resample=Image.BICUBIC) for image in stack]
    newMask = [image.resize((w_dash, h_dash), resample=Image.BICUBIC) for image in mask_stack]

    # crop can take negative = padding
    border = zoom_padding(xFactor, yFactor, w_dash, h_dash, w, h)

    newStack2 = [new.crop(border) for new in newStack]
    newMask2 = [new.crop(border) for new in newMask]

    return newStack2, newMask2

def rotateQuarter(stack, mask_stack):
    
    '''
    this function rotates images by 180, 270 or 90 degrees.
    
    Inputs:
    stack: a list of images
    mask_stack: a list of mask
    
    Outputs:
    newStack: a list of rotated images
    newMask: their masks
        
    '''
    
    mode = random.randint(0,3)

    if mode == 0:
        newStack = [image.transpose(Image.ROTATE_180)for image in stack]
        newMask = [image.transpose(Image.ROTATE_180) for image in mask_stack]
        
    elif mode == 1:
        newStack = [image.transpose(Image.ROTATE_90)for image in stack]
        newMask = [image.transpose(Image.ROTATE_90) for image in mask_stack]
    elif mode == 2:
        newStack = [image.transpose(Image.ROTATE_270)for image in stack]
        newMask = [image.transpose(Image.ROTATE_270) for image in mask_stack]
    else:
        newStack = stack
        newMask = mask_stack
    return newStack, newMask

def rotate(stack, mask_stack):
    
    '''
    this function rotates images by certain degree in integer.
    
    Inputs:
    stack: a list of images
    mask_stack: a list of mask
    
    Outputs:
    newStack: a list of rotated images
    newMask: their masks
        
    '''
    angle = np.random.randint(0, 360)
    newStack = [image.rotate(angle) for image in stack]
    newMask = [image.rotate(angle) for image in mask_stack]
    return newStack, newMask   

def flip(stack, mask_stack):
    
    '''
    this function flips images top to bottom or left or right.
    
    Inputs:
    stack: a list of images
    mask_stack: a list of mask
    
    Outputs:
    newStack: a list of flipped images
    newMask: their masks
        
    '''
    
    mode = random.randint(0,2)

    if mode == 0:
        newStack = [image.transpose(Image.FLIP_LEFT_RIGHT)for image in stack]
        newMask = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in mask_stack]

    elif mode == 1:
        newStack = [image.transpose(Image.FLIP_TOP_BOTTOM)for image in stack]
        newMask = [image.transpose(Image.FLIP_TOP_BOTTOM) for image in mask_stack]
    else:
        newStack = stack
        newMask = mask_stack
    return newStack, newMask


def cal_plane(skew,x1,x2,y1,y2, skew_amount):
    
    '''
    this function calculate the plane needed to skew the viewing angle.
    adapted from 
    https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
    
    Inputs:
    skew: the type of skew
    x1: one of the four corners of the original image plane
    x2: one of the four corners of the original image plane
    y1: one of the four corners of the original image plane
    y2: one of the four corners of the original image plane
    skew_amount: how much to skew
    
    Outputs:
    new_plane: new plane on which to project the new image
        
    '''
    
    if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":
        if skew == "TILT":
            skew_direction = random.randint(0, 3)
        elif skew == "TILT_LEFT_RIGHT":
            skew_direction = random.randint(0, 1)
        elif skew == "TILT_TOP_BOTTOM":
            skew_direction = random.randint(2, 3)

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),                # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1),                # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left

    if skew == "CORNER":

        skew_direction = random.randint(0, 7)

        if skew_direction == 0:
            # Skew possibility 0
            new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 1:
            # Skew possibility 1
            new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 2:
            # Skew possibility 2
            new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
        elif skew_direction == 3:
            # Skew possibility 3
            new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
        elif skew_direction == 4:
            # Skew possibility 4
            new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
        elif skew_direction == 5:
            # Skew possibility 5
            new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
        elif skew_direction == 6:
            # Skew possibility 6
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
        elif skew_direction == 7:
            # Skew possibility 7
            new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]
    
    return new_plane
    
def skew(stack, mask_stack, magnitude=0.5):
    '''
    this function skews the images by 0.5 which seems to be adequate.
    adapted from 
    https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
    
    Inputs:
    stack: a list of images
    mask_stack: a list of masks
    magnitude: how much to skew by
    
    Outputs:
    newStack: a list of transformed images
    newMask: their masks
    
    '''
    
    # take the first sample 
    image = stack[0]
    # Width and height taken from first image in list.
    w,h = image.size
    
    # original plane
    x1 = 0
    x2 = h
    y1 = 0
    y2 = w
    
    original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

    max_skew_amount = max(w, h)
    max_skew_amount = int(ceil(max_skew_amount * magnitude))
    skew_amount = random.randint(1, max_skew_amount)

    
    # We have two choices now: we tilt in one of four directions
    # or we skew a corner.
    options = ["TILT","TILT_LEFT_RIGHT","TILT_TOP_BOTTOM","CORNER"]
    
    new_plane = cal_plane(random.choice(options), x1,x2,y1,y2, skew_amount)

    
    # To calculate the coefficients required by PIL for the perspective skew,
    # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
    matrix = []
    for p1, p2 in zip(new_plane, original_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(original_plane).reshape(8)

    perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
    perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)
    
    newStack = [image.transform(image.size,Image.PERSPECTIVE,
                                   perspective_skew_coefficients_matrix,
                                   resample=Image.BICUBIC) for image in stack]
    newMask = [image.transform(image.size,Image.PERSPECTIVE,
                                   perspective_skew_coefficients_matrix,
                                   resample=Image.BICUBIC) for image in mask_stack]
    return newStack, newMask
    

def pillowFormat(x):
    
    '''
    this function helps with the conversion between PIL and numpy
    
    Inputs:
    x: input image channel (3) all identical, height, width, float64
    
    The problem is that we need the value to range between 0-1
    and in bytes
    
    Outputs:
    x2: image in PIL
    
    '''
    
    # my default colormaps for mask and images
    img_cmap = plt.cm.gray# check input type
    x = x - x.min()
    x = x / x.max()
    colors = Normalize(0, 1, clip=True)(x[0,:,:])
    x = img_cmap(colors, bytes=True)

    x2 = Image.fromarray(x)
    #x2.show() # for visualization
    return x2

def pillowFormatMask(mask):
    '''
    This function restricts array type and produce PIL formats
    
    Outputs need to be in the 255 range, not boolean True or False or 0-1
    
    Inputs: mask in numpy
    Outputs: mask in PIL
    
    '''
    mask = mask.astype(np.uint8) * 255
    mask2 = [Image.fromarray(mask[i,:,:]) for i in range(mask.shape[0])]
    return mask2
    
def erase(image):
    '''
    This function erases a patch of arbitrary dimensions on an image.
    
    Inputs:
    image: a single image
    
    Outputs:
    image: image with a patch
    
    '''
    
    # make sure we are using PIL format for this operation    
    if type(image) == np.ndarray:
        image = pillowFormat(image)
    
    w, h = image.size
    
    # percentage of the image
    # need clip at 0.15 otherwise we may get area smaller than 0.1/ minium
    rectangle_area = np.clip(np.random.rand(), 0.15,1) 

    w_occlusion_max = int(w * rectangle_area)
    h_occlusion_max = int(h * rectangle_area)

    w_occlusion_min = int(w * 0.1)
    h_occlusion_min = int(h * 0.1)

    w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
    h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

    rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
    
    random_position_x = random.randint(0, w - w_occlusion)
    random_position_y = random.randint(0, h - h_occlusion)

    image.paste(rectangle, (random_position_x, random_position_y))
    
    return image

def display(stack):
    
    '''
    this function shows one in every 10 slices of image stack.
    
    Inputs:
    stack: a list of images in PIL
    '''
    

    for img in stack[0:-1:10]:
        img.show()

def shear(stack, mask_stack):
    
    '''
    This function shears stack of images and their masks.
    adapted from 
    https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py
    
    Inputs:
    stack: a list of images
    mask_stack: their masks
    
    Outputs:
    newStack2: a list of sheared images
    newMask2: their masks
    
    '''
    
    # +-25 looks shear enough
    angle_to_shear = int(random.uniform((abs(24)*-1) - 1, 24 + 1))
    if angle_to_shear != -1: angle_to_shear += 1
    #print (angle_to_shear)

    
    directions = ["x", "y"]
    direction = random.choice(directions)
    
    # We use the angle phi in radians later
    phi = math.tan(math.radians(angle_to_shear))
    
    # holder of the transformed images
    newStack = []
    newMask = []
    
    image = stack[0]
    w,h = image.size

    if direction == "x":
        # Here we need the unknown b, where a is
        # the height of the image and phi is the
        # angle we want to shear (our knowns):
        # b = tan(phi) * a
        shift_in_pixels = phi * h

        if shift_in_pixels > 0:
            shift_in_pixels = math.ceil(shift_in_pixels)
        else:
            shift_in_pixels = math.floor(shift_in_pixels)

        # For negative tilts, we reverse phi and set offset to 0
        # Also matrix offset differs from pixel shift for neg
        # but not for pos so we will copy this value in case
        # we need to change it
        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        # Note: PIL expects the inverse scale, so 1/scale_factor for example.
        transform_matrix = (1, phi, -matrix_offset,
                            0, 1, 0)
        for image in stack:
            image = image.transform((int(round(w + shift_in_pixels)), h),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC)

            image = image.crop((abs(shift_in_pixels), 0, w, h))
            
            newStack.append(image)
        
        for mask in mask_stack:
            mask = mask.transform((int(round(w + shift_in_pixels)), h),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC)

            mask = mask.crop((abs(shift_in_pixels), 0, w, h))
            newMask.append(mask)

        

    elif direction == "y":
        shift_in_pixels = phi * w

        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        transform_matrix = (1, 0, 0,
                            phi, 1, -matrix_offset)
        
        for image in stack:
            image = image.transform((w, int(round(h + shift_in_pixels))),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC)

            image = image.crop((0, abs(shift_in_pixels), w, h))
            
            newStack.append(image)
            
        for mask in mask_stack:
            
            mask = mask.transform((w, int(round(h + shift_in_pixels))),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC)

            mask = mask.crop((0, abs(shift_in_pixels), w, h))
            newMask.append(mask)


        

    newMask2 = [image.resize((w, h), resample=Image.BICUBIC) for image in newMask]
    newStack2 = [image.resize((w, h), resample=Image.BICUBIC) for image in newStack]
    return newStack2, newMask2


def find_ndim_which_axis(image, axis, non_alpha_index):
    
    '''
    this function identifies the non alpha / non non transparency channel index.
    
    Inputs:
    image: an image in numpy
    axis: the dimension to be taken out
    non_alpha_index: dimensions other than transparency
    
    Outputs:
    image: image without alpha channel
    
    '''

    if axis == 0:
        image = image[non_alpha_index,:,:]
    elif axis == 2:
        image = image[:,:,non_alpha_index]
    return image
    
def plot_transformed(mask, image, name, output_dir):
    '''
    this function plots a single pair of image and its mask, so that debugging is possible directly.
    
    Inputs:
    mask: image's mask
    image: the MRI image
    name: file output
    output_dir: output directory
    
    
    '''

    # convert if the type is not numpy array
    if type(mask) != np.ndarray:
        mask = np.array(mask)
    if type(image) != np.ndarray:
        image = np.array(image)
    
    # ditch the useless channels
    if image.ndim  == 3:
        # check the order, c,h,w or h,w,c
        image_dim = image.shape
        color_channel = min(image_dim)
        axis = image_dim.index(color_channel)
        assert (axis == 0 or axis == 2),'image channel order may be wrong; color channel is at axis %d' %axis
        
        if  color_channel == 3 or color_channel == 4:
            ''' this would not work if there is a black border
            very problematic with zoom out. zoom in is okay
            the alpha is by default always the last one '''
            non_alpha_index = 0
            image = find_ndim_which_axis(image, axis,non_alpha_index)
        else:
            print ('color channel is %d not 3 or 4' %color_channel)
    # my default colormaps for mask and images
    mask_cmap = plt.cm.spring
    img_cmap = plt.cm.gray
    
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
    
    fig, ax_image = plt.subplots()
    ax_image.imshow(mask, cmap=mask_cmap)
    ax_image.imshow(colors, cmap=img_cmap)
    ax_image.set_axis_off()
    plt.savefig(os.path.join(output_dir, name))
    plt.close()


def save(image4, mask4, counter, output_dir) :
    ''' 
    This function creates a folder to save all the slices in the image stacks along
    with their masks
    
    Inputs:
    image4: a list of images
    masks4: a list of their masks
    counter: slice number
    output_dir: output folder
    
    
    '''
    target_path = os.path.join(output_dir, counter)
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    for subcounter, (image, mask) in enumerate(zip(image4, mask4)):
        name = counter+ '_' + str(subcounter)
        plot_transformed(mask, image, name, target_path)
 

def hist_match(source, template):
    """
    This function matches the intensity of a template image to a source image.
    
    taken from 
    https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    
    see images, source's intensity is tuned !
    
    
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image 
    """

    oldshape = source.shape
    # contiguous flattened array
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cum-sum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def imageReshape(img):
    '''
    this function reshape the image array to the right dimension.
    
    Inputs:
    img: numpy array image
    
    outputs:
    image5: numpy array image after reshape
    '''
    image5 = list(map(np.array, img)) # get uint8
    image5 = addAxis(image5)
    
    # check for color channels
    if image5.ndim == 4:
        # drop alpha channel
        image5 = image5[:,:,:,0:-1]

        # add axis
        image5 = image5[:, np.newaxis,:,:,:]

        image5 = np.swapaxes(image5, 4, 1)

        image5 = np.squeeze(image5)
    
    return image5

def toUint(image4):

    ''' 
    This function converts a list of PIL images to numpy array in uint8.
    
    Inputs:
    image4: list of PIL images
    
    Outputs:
    images: numpy array (depth, channel height, width) in uint8
    
    '''
    image5 = imageReshape(image4)
        
    return image5.astype(np.uint8)
    
def toFloat(image4):
    ''' 
    This function converts a list of PIL images to numpy array in uint8.
    
    Inputs:
    image4: list of PIL images
    
    Outputs:
    images: numpy array (depth, channel height, width) in float32
    
    '''
    image5 = imageReshape(image4)
        
    return image5.astype(np.float32)

def fromFloat_to_uint8(image6):
    ''' converting input (depth, channel, height, width) from 
    type float32 to output in uint8
    '''
    return image6.astype(np.uint8)
def addAxis(x):
    return  np.concatenate([m[np.newaxis,:,:] for m in x], axis=0)  

def gen(num):
    '''
    A generator that produce an infinite length of sequence of numbers.
    It is useful for training 
    
    Inputs:
    num: the total length of a list
    
    Outputs:
    an integer that denotes the particular slice of the data
    '''
    data = [i for i in range(num)]
    random.shuffle(data)
    yield from itertools.cycle(data)

def reverse_img_mask_pair(matched, mask):
    '''  input image format: list[items] length = depth 
     item = channels, height, width uint8 
     mask = array [depth, height, width]
     '''
    if np.random.rand() > 0.5:
        #print ('reverse')
        matched.reverse()
        mask = np.copy(mask[::-1,:,:])
        # don't forget this line; otherwise we get 
        # TypeError: 'NoneType' object is not iterable
        return matched, mask
    else:
        return matched, mask

def crop_resize(image3, mask, ratio):
    ''' input: list of PIL images, PIL images mask
    return PIL 
    '''
    w, h = image3[0].size
    # square box
    # not too tiny, divide the image into 4 quadrants
    max_x = int(w*ratio)
    max_y = int(h*ratio)
    x = random.randint(0, w-max_x-1) 
    y = random.randint(0, h-max_y-1) 
    #print("Cropping: {},{} -> {},{}".format(x,y, x+max_x, y+max_y)) 
    image4 = [im.crop((x,y, x+max_x, y+max_y)).resize((w, h), resample=Image.BICUBIC) for im in image3]
    #mask = pillowFormatMask(mask)
    mask2 = [im.crop((x,y, x+max_x, y+max_y)).resize((w, h), resample=Image.BICUBIC) for im in mask]
#    display(mask2)
#    display(image4)
    return image4, mask2
    
def operations(source_index, template_index, train_targets, train_images, output_dir):
    
    ''' 
    this function bundles a chain of operations that transform images at random.
    
    Inputs:
    source_index: the index of the source images for histogram matching
    template_index: the index of the template images for histogram matching
    train_targets: masks
    train_images: MIR images (samples, depth, channels, height, width) in uint8
    output_dir: save to this directory
    
    Outputs:
    image5: transformed image stacks
    mask5: their masks
        
    '''

    sources, templates = train_images[source_index,:,:,:], train_images[template_index,:,:,:]
    matched = [hist_match(source, template)  for source, template in zip(sources, templates)]
    mask = train_targets[source_index,:,:,:]

    image1, mask = reverse_img_mask_pair(matched, mask)
    
    
    image3 = [erase(img) for img in image1]
  
    
    
    counter = random.choice(['shear', 'zoom','rotateQuarter','rotate','flip','skew','crop','nothing'])

    if counter == 'shear':
        image4, mask4 = shear(image3, pillowFormatMask(mask))
    elif counter == 'zoom':
        image4, mask4 = zoom(image3, pillowFormatMask(mask))

    elif counter == 'rotateQuarter':
        image4, mask4 = rotateQuarter(image3, pillowFormatMask(mask))

    elif counter == 'rotate':
        image4, mask4 = rotate(image3, pillowFormatMask(mask))
        
    
    elif counter == 'flip':
        image4, mask4 = flip(image3, pillowFormatMask(mask))
    elif counter == 'skew':
        image4, mask4 = skew(image3, pillowFormatMask(mask))
    elif counter == 'crop':
        image4, mask4 = crop_resize(image3,  pillowFormatMask(mask), 2/3)
    else:
        image4, mask4 = image3, pillowFormatMask(mask)
    

    # save disk space by about 1/3 when Uint8 is used compared with float32 ! 
    image5 = toUint(image4)
    mask5 = toUint(mask4)

    return image5, mask5

def imageGen(images, masks, output_dir):
    '''
    this is a generator that yield a pair of transformed images and masks
    
    Inputs:
    images: one stack of images
    masks: their masks
    output_dir: not used here but pass it along
                (other functions use it to save output to this directory)
    
    Outputs:
    image5: transformed image stacks
    mask5: their masks
    '''
    
    num_samples = images.shape[0]

    # ensure never self-matching for histogram matching
    all_indices = []
    for i in range(num_samples):
        temp = [j for j in range(num_samples)]
        temp.pop(i)
        all_indices.append(temp)
    
    for source_index in gen(num_samples):
        template_index = random.choice(all_indices[source_index])
        image5, mask5 = operations(source_index, template_index, masks, images, output_dir)
        yield image5, mask5
    
      
if __name__ == '__main__':
    pass
