from PIL import Image
import numpy as np
import os, re, time, random

from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

try:
    from patch_dataloader.captcha.utils.helper import custom_augmentation, getAllFiles, make_dir
except:
    from captcha.utils.helper import custom_augmentation, getAllFiles, make_dir

# -------------------- # -------------------- # -------------------- # -------------------- # -------------------- # -------------------- #
class Dataset_patches(Dataset):
    '''
    __init__
        Inputs
            patch_dir : folder with .npy files containing patches
            label_dir : folder with .npy files containing either 
                            - an array of patch-wise labels | (shape = (number of patches))
                            - an array of 2D arrays of pixel-wise labeled patches | (shape = (number of patches, patch_size, patch_size))
            normalize : bool, if True, loaded patch tensors will be normalized
            pixel_wise : bool, if True, considers label_dir as contaning .npy of 2D arrays of pixel-wise labeled patches
                        else, 
                        either converts 2D array of pixel-wise labels to 1D array of patch wise labels
                        either, if the labels were already 1D array changes nothing
    __getitem__
        Output
            patch : if normalize
                        normalized tensor of size [1, patch_size, patch_size]
                    else
                        tensor of size [1, patch_size, patch_size]
            label : if pixel_wise
                        tensor of size [1,patch_size,patch_size] | 1 => vessel else 0 
                    else
                        tensor of size [1] | 1 => vessel else 0
    '''
    def __init__(self, dataset_dir, patch_size=96, normalize = False, augmentation = None, pixel_wise = False, current_ids = None, debug = False, saving_path = "png_dataset", load_empty = False):
        
        #Data Augmentation
        self.augmentation = augmentation
        self.transforms = custom_augmentation
        
        self.normalize = normalize
        patch_list = []
        label_list = []
        file_list = []
        
        means = []
        variances = []
        
        i = 0

        unfiltered_filelist = getAllFiles(dataset_dir)
        
        img_condition = re.compile('_img')
        label_condition = re.compile('_label')
        
        if current_ids is None: # If no train_samples is given, take all the patches
            patch_file_list = sorted([item for item in unfiltered_filelist if img_condition.search(item)])
            mask_file_list = sorted([item for item in unfiltered_filelist if label_condition.search(item)])
            
        else:
            # Filter the file list to only keep the patches and labels of the ID in the train samples
            train_samples_condition = lambda item: item.split('/')[-1].split('_')[0] in current_ids
            
            patch_file_list = sorted([item for item in unfiltered_filelist if img_condition.search(item) and train_samples_condition(item)])
            mask_file_list = sorted([item for item in unfiltered_filelist if label_condition.search(item) and train_samples_condition(item)])
        
        
        # DEBUG ----------------- # ----------------- # ----------------- # ----------------- #
        if debug:
            # Take only the first 3 patches of the dataset
            patch_file_list = patch_file_list[:3]
            mask_file_list = mask_file_list[:3]
        # ------------------ # ------------------- # ------------------- # ------------------ #
        assert len(patch_file_list) == len(mask_file_list), 'Number of patches and labels do not match'
        assert len(patch_file_list) >0, "No patches found in the given directory"
        
        # for patch, mask in zip(patch_file_list, mask_file_list):
        #     print(f"{patch = } <------> {mask = }")
        
        print(f"Loading {len(patch_file_list)} WNet patches...")
        
        print(f"Normalize: {self.normalize}, Augmentation: {self.augmentation}")
        
        start_loading = time.time()
        for patch_file,label_file in tqdm(zip(patch_file_list, mask_file_list), total=len(patch_file_list)):
            
            assert patch_file.split('/')[-1].split('_')[0] == label_file.split('/')[-1].split('_')[0], "Patch and label file names (ID) do not match"
            
            #Load patch and label
            patch = (np.load(os.path.join(dataset_dir,patch_file))) 
            label = (np.load(os.path.join(dataset_dir,label_file)))
            len_patch = patch.shape[0]
            
            if load_empty:
                patch_empty = (np.load(os.path.join(dataset_dir,patch_file).replace('true', 'empty'))) 
                label_empty = (np.load(os.path.join(dataset_dir,label_file).replace('true', 'empty')))
                
                patch = np.concatenate((patch, patch_empty), axis=0)
                label = np.concatenate((label, label_empty), axis=0)
                assert len_patch * 1.5 == patch.shape[0], f"Patch shape: {patch.shape = } | {len_patch = }"
            
            #print(patch_file)
            #print(label_file)
            
            # Iterate over each 96x96 patch, compute mean and variance for each one !!!
            
            # For approach
            #for img in patch:
                #curr_mean = np.mean(img)
                #means.append(curr_mean)
                #variances.append(np.mean((img - curr_mean)**2))
            
            # List comprehension approach
            if self.normalize:
                means.extend([np.mean(img) for img in patch])
                variances.extend([np.mean((img - np.mean(img))**2) for img in patch])
            
            if saving_path:
                # go back to the parent directory of the dataset_dir

                saving_path = os.path.join(os.path.dirname(os.path.normpath(dataset_dir)), saving_path)
                saving_path_patch = os.path.join(saving_path, "patches")
                saving_path_label = os.path.join(saving_path, "labels")
                
                make_dir(saving_path_patch)
                make_dir(saving_path_label)
                for n, (single_patch, single_label) in enumerate(zip(patch, label)):
                    assert single_patch.shape == single_label.shape, f"Patch and label shape do not match. {single_patch.shape = } | {single_label.shape = }"
                    if single_patch.shape != (32, 32):
                        print(f"Skipping {patch_file}, patch shape is not 32x32. {single_patch.shape = } {n}")
                        continue
                    assert single_patch.shape == (32, 32), f"Patch shape is not 32x32. {single_patch.shape = } for {patch_file.replace('.npy', '')}_{n}"
                    
                    if single_label.sum() == 0 and not load_empty:
                        print(f"Skipping {patch_file.replace('.npy', '')}_{n}.png")
                        print(f"Label sum: {single_label.sum()}")
                        print(f"Unique values: {np.unique(single_label)}")
                        assert(), "Label is empty"
                        continue
                    
                    # save as png
                    #patch_img = Image.fromarray(single_patch)
                    label_img = Image.fromarray(255*single_label.astype(np.uint8), 'L')
                    
                    #patch_img.save(os.path.join(saving_path_patch, f"{patch_file.replace('.npy', '')}_{n}.png"))
                    label_name = label_file.split('/')[-1].replace('.npy', '') + f"_{n}.png"
                    label_img.save(os.path.join(saving_path_label, label_name))
                    
            patch_list[i:i+patch.shape[0]] = patch[:]
            label_list[i:i+label.shape[0]] = label[:]
            file_list.append(patch_file)
            
            i += patch.shape[0]
        
        # ------------------ # ------------------- # ------------------- # ------------------ #
        # Given the mean and variance of each 96x96 patch in the training set,
        # We can compute the mean and std of the entire training set,
        # which will be used to normalize the test set
        if self.normalize:
            self.mean = np.mean(means)
            self.std = np.sqrt(np.mean(variances))
            print(f"Training metadata: Mean: {self.mean:.2f} | Std: {self.std:.2f}")
        else:
            print("No normalization applied")
            self.mean = 0
            self.std = 1
        
        print(f"Loading patches took {(time.time() - start_loading):.2f} seconds")
        # ------------------ # ------------------- # ------------------- # ------------------ #
        
        self.patches =  patch_list
        self.labels = label_list
        self.file_list = file_list
        self.pixel_wise = pixel_wise
    
    def __len__(self):
        return len(self.patches)
    
    def save_metadata(self, path):
        """
        Saves the mean and std of the training set to a file
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Saving metadata to {path}")
        #Save the mean and std of the training set
        metadata = {'mean': self.mean,'std': self.std}
        # save the dictionary to a file
        np.savez(os.path.join(path,'metadata.npz'), **metadata)
        print(f"Metadata saved (mean: {self.mean:.2f}, std: {self.std:.2f}) in {path}")    
    def __getitem__(self, idx):
        
        patch = torch.from_numpy(self.patches[idx]).float() # shape = [patch_size, patch_size]

        # Create a dummy dimension for the channels(=1) to get the conventional [B,C,H,W] tensors when using the dataloader 
        patch = patch[None, : , :] #shape = [ (batch_size,) 1, patch_size, patch_size]

        shape_pre, type_pre = patch.shape, patch.dtype
        
        if self.pixel_wise:
            label = torch.from_numpy(self.labels[idx]).float()
            label= label[None, : , :]  #shape = [ (batch_size,) 1, patch_size, patch_size]
        else:
            # creates a 1D array of size 1 with the value np.max(self.labels[idx]) (get patch wise regardless of the shape of the labels)
            label = torch.from_numpy(np.full((1), np.max(self.labels[idx]))).float() 
            print(f"Patch-wise Labels: {label.shape = }")
        
        shape_pre_label, type_pre_label = label.shape, label.dtype
        
        # Data Augmentation
        if self.augmentation and random.random() < self.augmentation: # 40% of the time, apply augmentation
            #print("Applying augmentation for both patch and label")
            patch, label = self.transforms(patch, label)
            assert patch.shape == shape_pre, f"Patch shape changed after augmentation. {shape_pre = } | {patch.shape = }"
            assert patch.dtype == type_pre, f"Patch type changed after augmentation. {type_pre = } | {patch.dtype = }"
            
            assert label.shape == shape_pre_label, f"Label shape changed after augmentation. {shape_pre_label = } | {label.shape = }"
            assert label.dtype == type_pre_label, f"Label type changed after augmentation. {type_pre_label = } | {label.dtype = }"
        
        # Data Normalization (Using the mean and std of the training set)
        if self.normalize:
        
            #print(f"Shape {patch.shape = }, Max {torch.max(patch) = }, Min {torch.min(patch) = }")
            
            #Standardize using the mean and std of the training set
            patch = Normalize(mean=[self.mean], std=[self.std])(patch)
            #Normalize between 0 and 1
            #patch = (patch - torch.min(patch))/(torch.max(patch) - torch.min(patch))
            
            #Normalize between -1 and 1
            #patch = (patch - torch.mean(patch))/torch.std(patch)
            
            #print(f"Shape {patch.shape = }, Max {torch.max(patch) = }, Min {torch.min(patch) = }\n")
        
        assert patch.shape == label.shape, f"Patch ({patch.shape}) and label ({label.shape}) size do not match"
        
        return patch, label
