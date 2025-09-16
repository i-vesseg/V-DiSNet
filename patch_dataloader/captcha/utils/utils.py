import os, re
import random
from tqdm import tqdm
import numpy as np

from itertools import zip_longest
import logging
from pathlib import Path
try:
    from patch_dataloader.captcha.utils.helper import getAllFiles, load_nifti_mat_from_file
except:
    from captcha.utils.helper import getAllFiles, load_nifti_mat_from_file
import torch
import torch.nn as nn
import torchvision.transforms as T


# ----------------------------------------------------------------------------------------------------------#
def compute_average_score(lines):
    total_score = sum(float(line.split(',')[1]) for line in lines)
    return total_score / len(lines)

def select_stochastic_batches(lines, n,m):
    
    """
    Shuffle the lines randomly and sort them by average score in descending order.
    Then, select the top m batches with the highest average scores.
    
    Args:
        lines (list): List of lines to be shuffled and sorted.
        
        [    "0  ), 1, 007_3201_3D_ToF_SENSE_3D_ToF.nii.gz, 62 , 2 , 2",
            "1  ), 2, 007_3201_3D_ToF_SENSE_3D_ToF.nii.gz, 61 , 2 , 2",
            "2  ), 3, 007_3201_3D_ToF_SENSE_3D_ToF.nii.gz, 12 , 3 , 4",
            "3  ), 4, 20_3201_3D_ToF_SENSE.nii.gz, 18 , 0 , 2",
            "4  ), 20, 007_3201_3D_ToF_SENSE_3D_ToF.nii.gz, 1  , 2 , 4",
            "5  ), 6, 007_3201_3D_ToF_SENSE_3D_ToF.nii.gz, 63 , 2 , 2"   ]  
    
        n (int): Number of lines in each batch (batch size).
        m (int): Number of batches to select (top m batches with the highest average scores).
    """
    
    random.shuffle(lines)  # Shuffle the lines randomly
    
    #print("\nShuffled lines:\n")
    #for line in lines:
    #    print(line)
    
    avg_scores = []  # List to store average scores
    
    for i in range(0, len(lines), n):
        
        batch_lines = lines[i:i+n]  # Get a batch of n lines
        
        #print(f"\nBatch {i//n+1}:\n")
        #for line in batch_lines:
        #    print(line)
        
        avg_score = compute_average_score(batch_lines)
        print(f"Average score: {avg_score}")
        avg_scores.append((avg_score, batch_lines))
    
    avg_scores.sort(key=lambda x: x[0], reverse=True)  # Sort by average score in descending order
    
    sorted_lines = [line for _, batch_lines in avg_scores for line in batch_lines]  # Flatten the sorted lines
    
    # Select the top m batches with the highest average scores
    m = m if m <= len(avg_scores) else len(avg_scores)
    sorted_lines = [line for _, batch_lines in avg_scores[:m] for line in batch_lines]
    
    return sorted_lines

# ----------------------------------------------------------------------------------------------------------#

def create_img_slices_dict(filename, columns=2):
    
    img_slices = {}
    with open(filename, 'r') as file:
        
        next(file) # Skip the header line
        
        for line in file:
            if columns == 2:
                img, slice_num = line.strip().split(',')
                
                slice_num = int(slice_num) # Convert the slice number to an integer
                # Check if the image is already a key in the dictionary
                if img in img_slices:
                    img_slices[img].append(slice_num) # Append the slice number to the existing list
                else:
                    img_slices[img] = [slice_num] # Create a new key with the slice number as a list
            
            elif columns == 4:
                img, slice_num, x_patch, y_patch = line.strip().split(',')

                # Convert the slice number, x_patch, and y_patch to integers
                slice_num = int(slice_num)
                x_patch = int(x_patch)
                y_patch = int(y_patch)

                # Check if the image is already a key in the outer dictionary
                if img in img_slices:
                    if slice_num in img_slices[img]: # Check if the slice number is already a key in the inner dictionary
                        img_slices[img][slice_num].append((x_patch, y_patch)) # Append the patch coordinates to the existing list
                    else:
                        img_slices[img][slice_num] = [(x_patch, y_patch)] # Create a new key with the patch coordinates as a list
                else:
                    img_slices[img] = {slice_num: [(x_patch, y_patch)]} # Create a new key with the slice number as a dictionary
                        
    return img_slices

def generate_and_save_sequential_patches(AL_img, patch_extraction_dir, patch_sizes, highest_entropies_and_variances_filepath=None, debug=False):
    print(f"Generating patches for {AL_img}")
    
    print(f"Saving patches to {patch_extraction_dir}")
    #print(f"patch_sizes: {patch_sizes}")
    
    if highest_entropies_and_variances_filepath is not None:
        with open(highest_entropies_and_variances_filepath, "r") as f:
            print(f"Gathering slices from {highest_entropies_and_variances_filepath}")
            for line in f:
                columns_num = len(line.split(","))    
                if columns_num == 2: # Slice mode [img_name, slice_num]
                    granularity = "slice"
                elif columns_num == 4: # Patch mode [img_name, slice_num, x_patch_num, y_patch_num]
                    granularity = "patch"    
                else:
                    raise ValueError(f"Expected 2 or 4 elements in line, got {columns_num}")
                break
            
        print(f"Granularity: {granularity} level")
        
        img_slices_dict = create_img_slices_dict(highest_entropies_and_variances_filepath, columns=columns_num)
        #print(f"img_slices_dict: {img_slices_dict}")
        
                
    # Get patient id from filename
    patient_name = AL_img.split("/")[-1].split(".")[0]
    patient_id = patient_name.split("_")[0]
    
    # make lists in dictionaries for each extracted patch size
    img_patches = {}  # dictionary to save image patches
    label_patches = {}  # dictionary to save label patches
        
    patient_list = sorted([item for item in getAllFiles(os.path.dirname(AL_img)) if re.search(f'{patient_id}', item.split("/")[-1].split("_")[0])]) # Get all files that contain the patient_id
    assert(len(patient_list) == 3), f"Expected 3 files, got {len(patient_list)}: {patient_list}"
    #print(f"patient_list: {patient_list}") # img, mask, vessel_mask
    
    for patch_size in patch_sizes:
        
        img_patches[str(patch_size)] = []
        label_patches[str(patch_size)] = []
        
        # TODO : Check this function 
        print(f'Loading image... {patient_list[0]}')
        
        img_name = os.path.basename(patient_list[0])
        if img_name not in img_slices_dict:
            print(f"Patient {patient_list[0]} is not in img_slices_dict")
            continue
            
        assert "_ToF" in patient_list[0], f"Expected img, got {patient_list[0]}"
        img_mat = load_nifti_mat_from_file(patient_list[0])
        #img_mat = reshape_with_patch_size(img_mat, patch_size)
        
        print(f'Loading mask...{patient_list[1]}')
        assert "_mask" in patient_list[1], f"Expected mask, got {patient_list[1]}"
        mask_mat = load_nifti_mat_from_file(patient_list[1])
        #mask_mat = reshape_with_patch_size(mask_mat, patch_size)
        
        print(f'Loading vessel mask...{patient_list[2]}')
        assert "_vessel" in patient_list[2], f"Expected vessel mask, got {patient_list[2]}"
        vessel_mask_mat = load_nifti_mat_from_file(patient_list[2])
        #vessel_mask_mat = reshape_with_patch_size(vessel_mask_mat, patch_size)
        
        assert img_mat.shape == mask_mat.shape == vessel_mask_mat.shape, f"Expected img_mat, mask_mat, vessel_mask_mat to have the same shape, got {img_mat.shape} == {mask_mat.shape} == {vessel_mask_mat.shape}"
        
        # prediction matrix
        prob_mat = np.zeros(img_mat.shape, dtype=np.float32)
        x_dim, y_dim, _ = prob_mat.shape
        
        # get the x, y and z coordinates where there is brain
        x, y, z = np.where(mask_mat)
        
        #print(f'x shape: {x.shape}')
        #print(f'y shape: {y.shape}')
        #print(f'z shape: {z.shape}')
        
        # get the z slices with brain
        z_slices = np.unique(z)
        
        current_nr_extracted_patches = 0
        
        # proceed slice by slice
        for l in tqdm(z_slices, leave=False, desc="Slices"):
            
            if l not in img_slices_dict[img_name]:
                #print(f"Slice {l} is not among the slices with highest entropy and variance")
                continue
            
            #print(f'Slice: #{l}')
            slice_vox_inds = np.where(z == l)
            
            # find all x and y coordinates with brain in given slice
            x_in_slice = x[slice_vox_inds]
            y_in_slice = y[slice_vox_inds]
            
            # find min and max x and y coordinates
            slice_x_min, slice_x_max = min(x_in_slice), max(x_in_slice) 
            slice_y_min, slice_y_max = min(y_in_slice), max(y_in_slice)

            # calculate number of predicted patches in x and y direction in given slice
            num_of_x_patches = int(np.ceil((slice_x_max - slice_x_min) / patch_size))
            num_of_y_patches = int(np.ceil((slice_y_max - slice_y_min) / patch_size))
            
            # print(f'num of x and y patches: ({num_of_x_patches},{num_of_y_patches})')

            # predict patch by patch in given slice
            for m in range(num_of_x_patches):
                for n in range(num_of_y_patches):
                    
                    # ---- Patch parameters ---- #
                    slice_num = l
                    x_patch_num = m
                    y_patch_num = n
                    # --------------------------- #
                    
                    #print(f"Slice: {slice_num}, x_patch_num: {x_patch_num}, y_patch_num: {y_patch_num}")
                    
                    if granularity == "patch" and (x_patch_num, y_patch_num) not in img_slices_dict[img_name][slice_num]:
                        #print(f"Patch ({x_patch_num}, {y_patch_num}) is not among the patches with highest entropy and variance")
                        continue
                    
                    
                    # ----------- DEBUGGING ---------------- # 
                    #Skip 95% of the patches in debug mode
                    if debug and np.random.rand() < 0.95:
                        continue
                    # --------------------------------------- #
                    
                    # find the starting and ending x and y coordinates of given patch
                    patch_start_x = slice_x_min + patch_size * m
                    patch_end_x = slice_x_min + patch_size * (m + 1)
                    patch_start_y = slice_y_min + patch_size * n
                    patch_end_y = slice_y_min + patch_size * (n + 1)
                    
                    # if the dimensions of the probability matrix are exceeded shift back the last patch
                    if patch_end_x > x_dim:
                        patch_end_x = slice_x_max
                        patch_start_x = slice_x_max - patch_size
                    
                    if patch_end_y > y_dim:
                        patch_end_y = slice_y_max
                        patch_start_y = slice_y_max - patch_size

                    # get the patch with the found coordinates from the image matrix
                    img_patch = img_mat[patch_start_x: patch_end_x, patch_start_y: patch_end_y, l]
                    label_patch = vessel_mask_mat[patch_start_x: patch_end_x, patch_start_y: patch_end_y, l]

                    img_patches[str(patch_size)].append(img_patch)
                    label_patches[str(patch_size)].append(label_patch)
                        
                    current_nr_extracted_patches += 1
    
    print(f'Saving extracted patches...')    
    
    # save extracted patches as numpy arrays
    for size in patch_sizes:
        assert len(img_patches[str(size)]) == len(label_patches[str(size)]), "Number of image patches and label patches are not equal"
        #print(f'Number of extracted patches: {current_nr_extracted_patches}')
        
        directory = patch_extraction_dir
        
        img_saving_path = os.path.join(directory,patient_id + '_' + str(size) + '_al_img.npy')
        label_saving_path = os.path.join(directory,patient_id + '_' + str(size) + '_al_label.npy')
        
        #print(f'Image patches saved to {img_saving_path}')
        np.save(label_saving_path, np.asarray(label_patches[str(size)])
                )
        #print(f'Label patches saved to {label_saving_path}')
        np.save(img_saving_path, np.asarray(img_patches[str(size)]))
    
    #print(f'Patches extracted and saved successfully')
    

def shuffle_list_per_patient(idxs_list, patch_per_patient=None, patient_ids=None, ratio_split=None):
    
    patient_sublists = []
    if patch_per_patient:
        n= patch_per_patient
        # Split the list into sublists of size n (patient-wise)
        patient_sublists = [idxs_list[i:i+n] for i in range(0, len(idxs_list), n)]
        
    elif patient_ids:
        # Split the list into sublists of size n (patient-wise)
        patient_sublists = [idxs_list[val:patient_ids[i + 1]] for i, val in enumerate(patient_ids) if i < len(patient_ids) - 1]
        
            
    # Shuffle patient-wise
    random.shuffle(patient_sublists)
    # print([(sublist[0], sublist[-1]) for sublist in patient_sublists]) # Uncomment to see the shuffled sublists (seed-check)
  

    # Concatenate the sublists back together
    if ratio_split:
        
        validation_patient_num = int(ratio_split*len(patient_sublists))
        training_patient_num = len(patient_sublists) - validation_patient_num

        print(f"Splitting the dataset (patient-wise) into {(1-ratio_split)*100}% training and {(ratio_split)*100}% validation")
        print(f"Total number of patients: {len(patient_sublists)} ({training_patient_num} Training - {validation_patient_num} Validation)")
        
        #print(f"Training Idx: {[(sublist[0], sublist[-1]) for sublist in patient_sublists[validation_patient_num:]]}")
        #print(f"Validation Idx: {[(sublist[0], sublist[-1]) for sublist in patient_sublists[:validation_patient_num]]}")
        
        train_shuffled_patient_list = [element for sublist in patient_sublists[validation_patient_num:] for element in sublist]
        val_shuffled_patient_list = [element for sublist in patient_sublists[:validation_patient_num] for element in sublist]
        assert len(train_shuffled_patient_list) + len(val_shuffled_patient_list) == len(idxs_list), f"Number of training {len(train_shuffled_patient_list)} and validation {len(val_shuffled_patient_list)} patients do not add up to the total number of patients {len(idxs_list)}"
        assert len(set(train_shuffled_patient_list).intersection(val_shuffled_patient_list)) == 0, "The two lists have common elements."
        
        return train_shuffled_patient_list, val_shuffled_patient_list
    
    else:
        shuffled_patient_list = [element for sublist in patient_sublists for element in sublist]
        
        return shuffled_patient_list


def merge_lists(list1, list2, n=3):
    """
    Merge two lists by taking n elements from each list in turn.
    Since we have (image, mask, vessel_mask) we want to merge them in the following way:
    (img1, mask1, vessel_mask1, image2, mask2, vessel_mask2, ...)
    (a1,b1,c1,a2,b2,c2,...) ==> In this case n=3 (default value)
    """
    result = []
    
    for i, (el1,el2) in enumerate(zip_longest(list1,list2)):
        
        if el1 is not None and el1 not in result:
            for j in range(n):
                result.append(list1[i+j])
                
        if el2 is not None and el2 not in result:
            for j in range(n):
                result.append(list2[i+j])

    return result

def enable_test_time_dropout(module):
    dropout_count = 0
    
    if isinstance(module, nn.Dropout):
        module.train()
        dropout_count += 1
    
    if isinstance(module, nn.Module):
        for submodule in module.children():
            dropout_count += enable_test_time_dropout(submodule)
    
    return dropout_count

def disable_test_time_batchnorm(module):
    batchnorm_count = 0
    
    if isinstance(module, nn.BatchNorm2d):
        print(f"Disabling test-time batchnorm for {module.__class__.__name__}")
        module.eval()
        batchnorm_count += 1
    
    if isinstance(module, nn.Module):
        for submodule in module.children():
            batchnorm_count += disable_test_time_batchnorm(submodule)
    
    return batchnorm_count

def create_logger(logger_name, log_file_path):
    """
    This function creates a logger with a file handler and allows the user to specify the output directory.
    """
    # Create log file path if it does not exist
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    # Ensure that the log file is empty
    open(log_file_path, 'w')
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the logging level
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)

    # Create a formatter and set the format for the handler
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

    return logger


# ---------------------- Augmentation: from TTA paper ---------------------- #
def random_augmentation(x, aug_dic=None, flip_axis=2, rotaxis0=2, rotaxis1=3, augmentation_list=[], type='img',  aug_gaussian_mean=0,  aug_gaussian_std=0):
    """ We do augmentation (flip, rotation, mult(0.9 - 1.1)
    :param x: a tensor of shape (#channels, x, y) or (#channels, x, y, z)
    :param aug_dic: augmentation dictionary (if given)
    :param flip_axis: tensor axis for flipping
    :param rotaxis0: tensor first axis of rotation
    :param rotaxis1: tensor second axis of rotation
    :param type: type of input ('img' or 'target'). If 'target', no jitter or blurring will be applied
    """
    if aug_dic is None:
        # We get params for number of flips (0 or 1) and number of rotations (0 ro 3)
        flip = torch.randint(0, 2, (1,)).item() if 'flip' in augmentation_list else 0
        num_rot = torch.randint(0, 4, (1,)).item() if 'rotation' in augmentation_list else 0
        
        # We define the same value for amount of brightness, contrast, saturation and hue jitter.
        # The factor will be uniformly from [max(0, 1 - value), 1 + value], 
        # except for hue value which will be chosen between  0.5 < =[-value, value] <= 0.5
        jitter = 0.5 if 'jitter' in augmentation_list else 0
        
        # We define the same value for kernel size and max sigma. 
        # Sigma will be chosen uniformly at random between (0.1, value)
        blur = 3 if 'blur' in augmentation_list else 1
        
        mean_gaussian = aug_gaussian_mean if 'gaussian_noise' in augmentation_list else 0
        std_gaussian = aug_gaussian_std if 'gaussian_noise' in augmentation_list else 0

        aug_dic = {'flip': flip,
                    'rot': num_rot,
                    'jitter': jitter,
                    'blur': blur,
                    'mean_gaussian': mean_gaussian,
                    'std_gaussian': std_gaussian
                    }
    else:
        flip = aug_dic['flip']
        num_rot = aug_dic['rot']
        
        # If it is a target image, there will be no jitter and bluring transformation
        jitter = 0 if type == 'target' else aug_dic['jitter']
        blur = 1 if type == 'target' else aug_dic['blur']
        mean_gaussian = 0 if type == 'target' else aug_dic['mean_gaussian']
        std_gaussian = 0  if type == 'target' else aug_dic['std_gaussian']

    # We apply the transformations
    x_aug = augment_data(x, flip=flip, n_rotation=num_rot, flip_axis=flip_axis, rot_axis0=rotaxis0, rot_axis1=rotaxis1,
                         jitter=jitter, blur=blur, mean_gaussian=mean_gaussian, std_gaussian=std_gaussian)

    return x_aug, aug_dic

def augment_data(img, flip=0, n_rotation=0, flip_axis=2, rot_axis0=2, rot_axis1=3, jitter=0, blur=1, mean_gaussian=0, std_gaussian=0):
    """
    We apply the given transformation (flip and rotation) on the input image
    :param flip: [0 or 1] flip applied as the initial transformation
    :param flip: [0, 1, 2, 3] number of rotations applied as the initial transformation
    :param jitter:  (same) value for amount of brightness, contrast, saturation and hue jitter.
                    The factor will be uniformly from [max(0, 1 - value), 1 + value], 
                    except for hue value which will be chosen between  0.5 < =[-value, value] <= 0.5
    :param blur: (same) value of kernel size and sigma for Gaussian blur. Kernel will have shape (value, value)
                 Sigma will be chosen uniformly at random between 0.1 and that value.
    """
    if flip != 0:
        img = torch.flip(img, [flip_axis])
        
    if n_rotation !=0:
        img = torch.rot90(img, n_rotation, [rot_axis0, rot_axis1])
    
    if jitter != 0:
        transform = T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter)
        img = transform(img)
        
    if blur != 1:
        transform = T.GaussianBlur(kernel_size=(blur,blur), sigma=(0.1, blur))
        img = transform(img)
        
    if mean_gaussian != 0 or std_gaussian != 0:
        transform = AddGaussianNoise(mean_gaussian, std_gaussian)
        img = transform(img)
        
    return img

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        tensor = tensor.detach().cpu()
        noise = torch.randn(tensor.size())
        return tensor + noise * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
# --------------------------------------------------------------------------- #