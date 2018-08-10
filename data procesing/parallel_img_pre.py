import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from transform_HPC import imageGen
from mpi4py import MPI
from enhanceContrast_HPC import resize

def cutSliceBySlice(image, mask, size):
    """
    resize the stack of images to a new size
    
    inputs:
        images = array of images
        mask = their mask
        size = tuple of new dimensions
    
    outputs:
        numpy array of resized images and their mask
    """
    
    depth_kernels = image.shape[0]
    num_kernels = int(depth_kernels / depth)
    
    # must upcast otherwise code won't work
    image = image.astype(np.float64)
    mask = mask.astype(np.int64)
    
    image = image / image.max()
    
    mask_list = []
    img_list = []
    for k in range(num_kernels):
        for counter in range(depth):
            # single channel and upcast to float64
            image_slice = image[counter+k*depth,0,...]
            #print ('image slice shape {}'.format(image_slice.shape))
            
            img, img_pil = resize(image_slice, (height, width), img_cmap, isImage=True)
            img_list.append(img)
            
            if k ==0: # only work on the mask once regardless how many kernels are in use
                mask_slice = mask[counter+k*depth,...]
                
                mask_slice, _ = resize(mask_slice, (height,width), mask_cmap, isImage=False)
                mask_list.append(mask_slice)
    return np.asarray(img_list), np.asarray(mask_list)
            

def augment_images(train_images, train_targets, debug_dir, r, tag, MAX_SIZE, base_rank):
    
    """
    this function puts all the parts for later image augmentation
    
    inputs:
        train_images = image array
        train_targets = mask
        debug_dir = save to disk for debug
        r = ID code
        tag = training , validation or testing dataset
        MAX_SIZE = limit on the number of new images
        base_rank = reference code
     outputs:
         images and their mask in its original and lower resolutions
    """
    
    r = r+base_rank # keep track of IDs
    print('handling %s' %tag)
    print ('rank %d is working' %r)
    
    # use a generator
    generator_training = imageGen(train_images, train_targets, debug_dir)
    
    stack_img = []
    stack_masks = []
    th_stack_img = []
    th_stack_masks = []
    for i, (image, mask) in enumerate(generator_training):

        # save only a single channel
        sc_img = image[:,0,:,:]
        
        th_img, th_mask = cutSliceBySlice(image, mask, (height, width))
        

        flat = mask.flatten()
        ind = np.nonzero(flat)[0] # save only the locations of non-zero entries
        
        flat = th_mask.flatten()
        th_ind = np.nonzero(flat)[0]
        
        stack_img.append(sc_img[np.newaxis,...])
        stack_masks.append(ind[np.newaxis,...])
        
        th_img = th_img[:,0,:,:]
        th_stack_img.append(th_img[np.newaxis,...])
        th_stack_masks.append(th_ind[np.newaxis, ...])
        
        # need to minus 2 here !
        if i > (MAX_SIZE-2):
            break
    
    stack_img = np.vstack(stack_img)
    th_stack_img = np.vstack(th_stack_img)
    pickle.dump( (stack_masks, th_stack_masks), open( tag+"Rank_pairs"+str(r)+"_masks.p", "wb" ) )
    
    np.savez_compressed(os.path.join(training_dir, tag+'Rank_pairs'+str(r)),
                        images=stack_img,
                        th_images=th_stack_img)
    


if __name__ == '__main__':
    
    training_dir = r'/hpc/wfok007/mpi_heart'
    debug_dir = r'/hpc/wfok007/mpi_heart/debugPairs'
    fileName = 'pairs_input_data2.npz'
    npzfile = np.load(os.path.join(training_dir, fileName))
    train_images = npzfile['train_images']
    train_targets = npzfile['train_targets']
    valid_images = npzfile['valid_images']
    valid_targets = npzfile['valid_targets']
    test_images = npzfile['test_images']
    test_targets = npzfile['test_targets']
    
    print (train_images.shape)
    print (train_images.dtype)
    print ('min %f' %np.min(train_images))
    print ('max %f' %np.max(train_images))
    
    # my default colormaps for mask and images
    mask_cmap = plt.cm.spring
    img_cmap = plt.cm.gray
    
    # downsampling dimensions
    height = 112
    width = 112
    depth = 88
    
    MAX_SIZE = 30
    total_samples = 3000
    cpu = 25
    

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # prepare validation and testing datasets
    base_rank = 0
    augment_images(valid_images, valid_targets, debug_dir, rank, 'valid2', MAX_SIZE, base_rank)
    augment_images(test_images, test_targets, debug_dir, rank, 'test2', MAX_SIZE, base_rank)
    
    # Because we need more training data, repeat the process a few times
    repeats = total_samples/(cpu * MAX_SIZE )

    for base_rank in range(0, int(repeats*cpu+1), cpu):
        print (base_rank)
        augment_images(train_images, train_targets, debug_dir, rank, 'train2', MAX_SIZE, base_rank)
