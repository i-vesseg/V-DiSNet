"""
File name: helper.py
Author: ngocviendang
Date created: July 13, 2020
This file contains helper functions for other scripts.
"""
import os, random, re
from pathlib import Path
import nibabel as nib
import numpy as np
import subprocess

import torch
import torchvision.transforms as transforms
import heapq # heapq is a min-heap implementation (https://docs.python.org/3/library/heapq.html)


patch_per_patient_wnet = 1000 # number of patches to be extracted from each patient in WNetSeg


def make_dir(dir_path):
    """ 
        Create a directory if it does not exist.
    """
    assert dir_path is not None, f'Invalid dir path: {dir_path}'
    if isinstance(dir_path, list):
        print(f"WARNING: Creating multiple directories")
        for path in dir_path:
            Path(path).mkdir(parents=True, exist_ok=True)
    else:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    """
    Set the seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    # helps ensure deterministic behavior even when using CUDA for GPU acceleration.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_pairwise_lists(input_list, mask_list, label_list=None, label_list_2=None):
    
    if label_list is None:
        label_list = [None] * len(input_list)
    
    if label_list_2 is None:
        label_list_2 = [None] * len(input_list)
    
    for input, mask, label, label_2 in zip(input_list,mask_list,label_list, label_list_2):

        print(input)
        print(mask)
        if label is not None:
            print(label)
        if label_2 is not None:
            print(label_2)
        
        assert get_id_item(input) == get_id_item(mask), f'Id mismatch between input and mask: {input} vs {mask}'
        
        if label is not None:
            assert get_id_item(input) == get_id_item(label), f'Id mismatch between input and label: {input} vs {label}'
        if label_2 is not None:
            assert get_id_item(input) == get_id_item(label_2), f'Id mismatch between input and label_2: {input} vs {label_2}'
        
        
        
            
def get_id_item(item):
    item = item.replace('.nii.gz', '')
    item_name = os.path.basename(item)
    
    id = item_name.split('_')[1] if item_name.startswith('prediction') else item_name.split('_')[0]
    
    assert id.isdigit(), f'Id is not a digit: {id} for item: {item}'
    return id

def appeand_to_heap(heap_file, annotation_number, uncertainty, patient_filename, l, patch_position):    
    if len(heap_file) < annotation_number:
        heapq.heappush(heap_file, (uncertainty, (patient_filename, l, patch_position))) # Add the new element to the heap (until we have n elements)
    else:
        heapq.heappushpop(heap_file, (uncertainty, (patient_filename, l, patch_position))) # Remove the smallest element and add the new one (if it's bigger)

def getAllFiles(dir, result = None, sort = False):
    if result is None:
        result = []
    for entry in os.listdir(dir):
        entrypath = os.path.join(dir, entry)
        if os.path.isdir(entrypath):
            getAllFiles(entrypath ,result)
        else:
            result.append(entrypath)
    if sort:
        result = sorted(result)
    return result

def getAllImgs(original_data_dir, regex_filter = "_img|_ToF"):
    return [item for item in getAllFiles(original_data_dir) if re.search(regex_filter, item)]

def load_nifti_mat_from_file(path_orig):
    """
    # Reference https://github.com/prediction2020/unet-vessel-segmentation.git
    Loads a nifti file and returns the data from the nifti file as numpy array.
    :param path_orig: String, path from where to load the nifti.
    :return: Nifti data as numpy array.
    """
    nifti_orig = nib.load(path_orig)

    print(' - nifti loaded from:', path_orig)
    print(' - dimensions of the loaded nifti: ', nifti_orig.shape)
    print(' - nifti data type:', nifti_orig.get_data_dtype())
    
    return nifti_orig.get_fdata()  # transform the images into np.ndarrays


def create_and_save_nifti(mat, path_target):
    """
    # Reference https://github.com/prediction2020/unet-vessel-segmentation.git
    Creates a nifti image from numpy array and saves it to given path.
    :param mat: Numpy array.
    :param path_target: String, path where to store the created nifti.
    """
    new_nifti = nib.Nifti1Image(mat, np.eye(4))  # create new nifti from matrix
    nib.save(new_nifti, path_target)  # save nifti to target dir
    print(f' - New nifti saved to: {path_target}')

def Rand(start, end, num):
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    return res

def get_brain_mask(img_path):
    
    assert img_path.endswith('.nii.gz'), f'Invalid mat path: {img_path}'
    assert os.path.exists(img_path), f'Image does not exist: {img_path}'
    
    brain_mask_path = img_path.replace('.nii.gz', '_brain_mask.nii.gz')
    skull_stripped_path = img_path.replace('.nii.gz', '_brain.nii.gz')
    
    # ----------------------------------------------------------------------------------------------------- #
    
    print(' - Creating brain mask using freesurfer')
    # $ mri_synthstrip -i mat_path -o output.nii.gz>/dev/null -m brain_mask.nii.gz
    
    # You always have to run the following commands before running mri_synthstrip
    
    #export FREESURFER_HOME=/data/falcetta/freesurfer # path to freesurfer
    #source $FREESURFER_HOME/SetUpFreeSurfer.sh # set up freesurfer

    print(' - Extracting brain mask')
    
    command = [
        "/data/falcetta/freesurfer/bin/mri_synthstrip",
        "-i", img_path,
        "-o", skull_stripped_path,
        "-m", brain_mask_path,
        "--gpu"
    ]
    
    
    run_bash_command(command)

    # ----------------------------------------------------------------------------------------------------- #
    assert os.path.exists(brain_mask_path), f'Brain mask does not exist: {brain_mask_path}'   
    brain_mat = load_nifti_mat_from_file(brain_mask_path)
    # remove the brain mask file
    os.remove(brain_mask_path)
    os.remove(skull_stripped_path)
    
    return brain_mat

def run_bash_command(command):
    try:
        print("Executing command: " + " ".join(command))
        subprocess.run(command, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        raise e
    
def apply_mask(mat, mask_mat, bg_value=0):
    """
    # Reference https://github.com/prediction2020/unet-vessel-segmentation.git
    Masks the image with the given mask.
    :param mat: Numpy array, image to be masked.
    :param mask_mat: Numpy array, mask.
    :return: Numpy array, masked image.
    """
    print(f' - Applying mask with background value: {bg_value}')
    masked = mat
    masked[np.where(mask_mat == 0)] = bg_value
    return masked

def format_duration(duration):
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    elif seconds > 0:
        return f"{seconds:.2f}s"
    else :
        return f"{duration:.2f}s"

def torch_argmax(tensor, dim=1):
    """
    Returns the indices of the maximum values of a tensor across a dimension.
    
    Inputs:
        tensor: A pytorch tensor. default dimension [batch_size, num_classes, height, width]
        dim: The dimension to apply argmax.
    """
    return torch.argmax(tensor, dim, keepdim=True)


def torch_one_hot(tensor, n_classes=2, device=None, task='Classification'): #PNET
    
    old = tensor
    
    if n_classes:
        if tensor.shape[0]>8: # given a small batch size, the tensor.max() may not be the max value in the tensor
            assert n_classes == tensor.max() + 1, f"num_classes ({n_classes}) should be equal to max value in tensor + 1 ({int(tensor.max())} + 1)"
    else:
        n_classes = tensor.max() + 1
        
    if device is None:
        device = tensor.device
            
    if task == 'Classification': #PNET
        """
        Returns a one-hot encoded tensor.
        
        Inputs:
            tensor: A pytorch tensor. default dimension [batch_size, num_classes]
            n_classes: Number of classes to apply one-hot encoding.
            device: The device to store the tensor.
        """    
        one_hot = torch.zeros((tensor.size(0), n_classes), device=device, dtype=torch.float)
        one_hot.scatter_(dim=1, index=tensor.long(), value=1)
    else: # task == 'Segmentation' (WNET)
        """
        Returns a one-hot encoded tensor. [batch_size, n_classes, height, width]
        
        Inputs:
            tensor: A pytorch tensor of shape [batch_size, 1, height, width]
            n_classes: The number of classes in the segmentation mask.
        """
        # input tensor of size [batch, channels, patchsize, patchsize]
        one_hot = torch.zeros((tensor.shape[0], n_classes, tensor.shape[2], tensor.shape[3]), device=device, dtype=torch.float)
        # output tensor of size [batch, n_classes, patchsize, patchsize]
        one_hot = one_hot.scatter(dim=1, index=tensor.long(), value=1)
    
    return one_hot 

def compute_uncertainty(prediction, method='entropy'):
    # prediction: [n_classes, height, width]
    
    assert method in ['entropy', 'max_prob'], f'Invalid uncertainty method: {method}'
    assert prediction.ndim == 3, f'Invalid prediction dimension: {prediction.ndim}, should be 3'
    assert prediction.shape[0] > 1, f'Invalid prediction shape: {prediction.shape}, should be [n_classes, height, width]'
    
    if method == 'entropy': # Compute entropy of the prediction per pixel 
        per_pixel_uncertainty = - torch.sum(prediction * torch.log(prediction + 1e-6), dim=0) # [height, width]
    
    elif method == 'max_prob': # Consider the maximum probability of the prediction per pixel
        per_pixel_uncertainty = 1 - torch.max(prediction, dim=0)[0] # [height, width]
        
    uncertainty = torch.mean(per_pixel_uncertainty) # Compute the (scalar) uncertainty by averaging the uncertainty per pixel
    
    return uncertainty # The higher the uncertainty, the more information you can gain by inserting this patch to the training set

def apply_test_time_augmentation(img_model_input, label_patch=None, patch_size=96):
    augmentation = custom_augmentation
    if label_patch is None:    
        img_model_input = augmentation(img_model_input)
        return img_model_input
    else :
        img_model_input, label_patch = augmentation(img_model_input, label_patch)
        return img_model_input, label_patch

def custom_augmentation(image, label=None, affine_transform=True, horizontal_flip=True, vertical_flip=True):
    
    if label:
        assert image.shape == label.shape, f'Invalid image and label shape: {image.shape} vs {label.shape}'
    
    old_shape = image.shape
    # Random affine transformation
    if affine_transform:
        affine_transform = transforms.RandomAffine(degrees=(-30, 30))
        image = affine_transform(image)
        if label:
            label = affine_transform(label)
        
    # Random horizontal flip
    if horizontal_flip and torch.rand(1).item() < 0.5:
        image = transforms.functional.hflip(image)
        if label:
            label = transforms.functional.hflip(label)
            
    # Random vertical flip
    if vertical_flip and torch.rand(1) < 0.5:
        image = transforms.functional.vflip(image)
        if label:
            label = transforms.functional.vflip(label)

    assert image.shape == old_shape, f'Invalid image shape: {image.shape} vs {old_shape}'
    
    if label:
        assert image.shape == label.shape, f'Invalid image and label shape: {image.shape} vs {label.shape}'
        return image, label
    
    return image

def standardizeNpVolume(image, mean=None, std=None):
    """
    Standardize the image volume by subtracting the mean and dividing by the standard deviation.

    Inputs:
        image: A numpy array of shape [height, width, depth] or [height, width, depth, channels]
        mean: The mean value to subtract from the image. If None, the mean of the image will be computed.
        std: The standard deviation to divide the image by. If None, the standard deviation of the image will be computed.
    """
    mean, std = (mean, std) if mean or std else (np.mean(image), np.std(image)) 
    image -= mean
    image /= std + 1e-8
    return image

def extract_roi(id, modality):
    """
    Open the corresponding roi file and filter out the roi with the given id.
    
    Inputs:
        label mat: A numpy array of shape [height, width, depth]
        id: The id of the roi to be extracted.
        modality: The modality of the image. Either 'ct' or 'mr'.
        
    Outputs:
        label_mat_roi: A numpy array of shape [height, width, depth] with the roi extracted. (0 everywhere else)
        size: The size of the roi.
        location: The location of the roi.
    """
    assert modality in ['ct', 'mr'], f'Invalid modality: {modality}, must be either ct or mr'
    
    
    print(f'\nExtracting roi with id: {id} and modality: {modality}')
    
    # ----------------------------------------------------------------------------------------------------- #
    all_roi_path = '/data/falcetta/TopCoW_Data_MICCAI2023/topcow_batch-TOT_PREPROCESSED/roi_size_loc'
    # ----------------------------------------------------------------------------------------------------- #
    
    assert os.path.exists(all_roi_path), f'Path to all roi files does not exist: {all_roi_path}'
    
    roi_files = getAllFiles(all_roi_path) # get all roi files
    assert len(roi_files) > 0, f'No roi files found in {all_roi_path}'
    
    roi_txt_files = [item for item in roi_files if item.endswith('.txt')] # get all roi txt files
    
    roi_txt_file_id = [item for item in roi_txt_files if f'_{id}' in item and modality in item] # get the roi txt file with the given id
    
    assert len(roi_txt_file_id) == 1, f'Found {len(roi_txt_file_id)} roi txt files with id {id}'
    roi_txt_file_id = roi_txt_file_id[0]
    
    print(f'Found roi txt file: {roi_txt_file_id}')
    
    with open (roi_txt_file_id, 'r') as f:
        """
            --- ROI Meta Data ---

            Size (Voxels): 87 60 26

            Location (Voxels): 86 85 94
        """
        # return the roi size and location
        for line in f:
            
            if line.startswith('Size'):
                size = [int(item) for item in line.split(':')[-1].strip().split(' ')]
                
                assert len(size) == 3, f'Invalid size: {size}'
                assert size[0] > 0 and size[1] > 0 and size[2] > 0, f'Invalid size: {size}'
            
            elif line.startswith('Location'):
                location = [int(item) for item in line.split(':')[-1].strip().split(' ')]
                
                assert len(location) == 3, f'Invalid location: {location}'
                assert location[0] >= 0 and location[1] >= 0 and location[2] >= 0, f'Invalid location: {location}'
                
        print(f'ROI size: {size}')
        print(f'ROI location: {location}')
        
        x_min, x_max = location[0], location[0] + size[0]
        y_min, y_max = location[1], location[1] + size[1]
        z_min, z_max = location[2], location[2] + size[2]
        
        return x_min, x_max, y_min, y_max, z_min, z_max