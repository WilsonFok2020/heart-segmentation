import numpy as np
import SimpleITK as sitk
import os

import matplotlib
matplotlib.use('agg')
#import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 8

def trim(x):
    """
    This function trims the edges off images.
    Input:
        x = stack of images (88x640x640 or 88x576x576)
    output:
        x = 1x88x576x576
    """
    # make sure we get a 3D stack not 2D slice
    assert (x.shape) != 3
    if x.shape[-1] > 576:
        newx = x[:,32:-32, 32:-32]
    else:
        newx = x
    return newx[np.newaxis,...]
    
def getFilePaths():
    
    """
    walk through the image directory to get all the image files
    
    Output:
        full file paths of images and their masks
    """
    
    image_dir = r'/hpc/wfok007/mpi_heart/Training Set'
    mask_paths = []
    image_paths = []
    for root, dirs, files in os.walk(image_dir, topdown=False):
        for name in files:
            if name == 'laendo.nrrd':
               mask_paths.append(os.path.join(root, name))
            elif name == 'lgemri.nrrd':
                image_paths.append(os.path.join(root, name))
            else:
                print ('%s is unknown' %name)
    return mask_paths, image_paths
    
def load_nrrd(full_path_filename):
    
    """
    This function loads and decode the raw image files.
    Input:
        full path of the image file
    Output:
        image array
    """
    data = sitk.ReadImage( full_path_filename )
    data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )
    data = sitk.GetArrayFromImage(data)
    return(data)


if __name__ == '__main__':
    
    # get file paths
    mask_paths, image_paths = getFilePaths()
    #handling the masks
    masks = [trim(load_nrrd(path)) for path in mask_paths]
    masks2 = [mask == mask.max() for mask in masks]
    masks3 = np.concatenate(masks2, axis=0)
    # handling the images
    images = [trim(load_nrrd(path)) for path in image_paths]
    images2 = np.concatenate(images, axis=0)
    

    training_dir = os.getcwd()
    np.savez_compressed(os.path.join(training_dir, 'input_data'),
                        masks=masks3,
                        images = images2)
