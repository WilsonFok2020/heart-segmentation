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
    
# this function loads .nrrd files into a 3D matrix and outputs it
# 	the input is the specified file path to the .nrrd file
def load_nrrd(full_path_filename):

	data = sitk.ReadImage( full_path_filename )
	data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )
	data = sitk.GetArrayFromImage(data)

	return(data)

def checkMask(data):
    ''' data = mask stack depth, width, height
    tell me if there is only 1 value, 255
    
    each heart has a different mask sizes, of course
    '''
    # only 0, 255 ? not so
    five = np.where(data == 255)
    for f in five:
        print (f.shape)
    
    zero = np.sum(data)
    return int(five[0].shape[0]) * 255 == zero
	
# this function encodes a 2D file into run-length-encoding format (RLE)
# 	the input is a 2D binary image (1 = positive), the output is a string of the RLE
def run_length_encoding(input_mask):
	
	dots = np.where(input_mask.T.flatten()==1)[0]
	
	run_lengths,prev = [],-2
	
	for b in dots:
		if (b>prev+1): run_lengths.extend((b+1, 0))

		run_lengths[-1] += 1
		prev = b

	return(" ".join([str(i) for i in run_lengths]))
	


### a sample script to produce a prediction 

# load the image file and reformat such that its axis are consistent with the MRI


if __name__ == '__main__':
    
    mask_paths, image_paths = getFilePaths()
    masks = [trim(load_nrrd(path)) for path in mask_paths]
    checking = [checkMask(mask) for mask in masks]
    masks2 = [mask == mask.max() for mask in masks]
    masks3 = np.concatenate(masks2, axis=0)
    
    
    
    # sparse matrix is 2D only
    #from scipy.sparse import csc_matrix
    #masks = [csc_matrix(mask) for mask in masks]
        
    images = [trim(load_nrrd(path)) for path in image_paths]
    images2 = np.concatenate(images, axis=0)
    
    #
    #
    training_dir = os.getcwd()
    np.savez_compressed(os.path.join(training_dir, 'input_data'),
                        masks=masks3,
                        images = images2)
    
    # check input dimension
#    for mask, image in zip(masks, images):
#        print ('{} \t {}'.format(mask.shape, image.shape))
#    # imageJ counts from 1
#    image = images[0][0,-1,...]












#
#
#
## *** your code goes here for predicting the mask:
#
#mask = np.zeros(image.shape)
#mask[image>200] = 1 # a very trivial solution is presented
#
## ***
#
#
#
## encode in RLE
#image_ids = ["ExampleOnlyMRI_slice_"+str(i) for i in range(image.shape[0])]
#
#encode_cavity = []
#for i in range(mask.shape[0]):
#	encode_cavity.append(run_length_encoding(mask[i,:,:]))
#
## output to csv file
#csv_output = pd.DataFrame(data={"ImageId":image_ids,'EncodeCavity':encode_cavity},columns=['ImageId','EncodeCavity'])
#csv_output.to_csv("ExampleOnlyLabels.csv",sep=",",index=False)