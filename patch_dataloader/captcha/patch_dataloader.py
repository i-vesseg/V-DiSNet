import os
import numpy as np

from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

try: 
    from patch_dataloader.captcha.utils.utils import shuffle_list_per_patient
    from patch_dataloader.captcha.utils.helper import patch_per_patient_wnet
    from patch_dataloader.captcha.get_data.get_p_data import Dataset_patches
except:
    from captcha.utils.utils import shuffle_list_per_patient
    from captcha.utils.helper import patch_per_patient_wnet
    from captcha.get_data.get_p_data import Dataset_patches

from torch.utils.data import Sampler

class FixedSubsetSampler(Sampler):
    r"""Gives a sampler that yields the same set of indices.

        Arguments:
            indices (sequence): a sequence of indices
        """
    def __init__(self, indices):
        self.idx = indices

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

def get_dataloader(dataset_dir=None, train_metadata_filepath=None,
               datetime_path=None, current_ids=None,
               patch_size = 96, validation_split = 0.2,batch_size = 200,
               normalize = True, augmentation = False ,pixel_wise = True, debug=False, n_test_samples=None, load_empty=False):
    
    #DATA RETRIVAL
    assert dataset_dir is not None, 'Dataset directory is not provided'
    
    #Create a unique path for each training
    datetime_path = datetime.now().strftime("%Y-%m-%d/%H-%M-%S") if datetime_path is None else datetime_path
    train_metadata_filepath = os.path.join(train_metadata_filepath)
    
    #MODEL, CRITERION & OPTIMIZER#
    
    print("Loading dataset for Wnet...")
    data = Dataset_patches(dataset_dir,patch_size, normalize, augmentation, pixel_wise, current_ids=current_ids, debug=debug, load_empty=load_empty)
    
    # Save train mean and std for normalization
    data.save_metadata(path=train_metadata_filepath)
        
    # Print model summary
    dataset_size = data.__len__()
    print(f"Dataset size: {dataset_size:,}")
    
    #TRAIN/VALIDATION SPLIT#    
    print(f"Train / Validation split...")
    idxs = list(range(dataset_size)) # 0 to dataset_size-1 (1000 per patient)
    patch_per_patient = patch_per_patient_wnet # This value is defined in captcha/preprocessing/seg_patch_extraction.py file
    
    if load_empty:
        patch_per_patient = int(patch_per_patient_wnet * 1.5)
    
    assert dataset_size % patch_per_patient == 0, f"Dataset size ({dataset_size}) must be a multiple of patch_per_patient ({patch_per_patient})"
    
    print(f"Patch per patient: {patch_per_patient}")
    
    # Shuffle the indices
    print(f"Shuffling {len(idxs)} indices patient-wise...")
    idxs = shuffle_list_per_patient(idxs, patch_per_patient)
    
    num_patient = int(dataset_size / patch_per_patient)
    
    print(f"Train-Validation percentage: {100*validation_split}%")
    
    validation_patients = int(np.ceil(validation_split * num_patient)) if validation_split > 0 else 0
    print(f"Tot number of patients: {num_patient} ({num_patient - validation_patients} Training - {validation_patients} Validation)")
    
    
    split = patch_per_patient * validation_patients
    train_idxs ,val_idxs = idxs[split:], idxs[:split]
    
    if n_test_samples is not None:
        val_idxs = val_idxs[:n_test_samples]

    assert len(set(train_idxs).intersection(val_idxs)) == 0, "The two lists have common elements."
        
    print(f"Train size: {len(train_idxs)} - Validation size: {len(val_idxs)}")

    #SAMPLER & DATALOADER#
    print(f"Creating dataloaders...")
    train_sampler = SubsetRandomSampler(train_idxs)
    val_sampler = FixedSubsetSampler(val_idxs) if len(val_idxs) > 0 else None
    

    train_dataloader = DataLoader(data, batch_size=batch_size, num_workers=4,
                            pin_memory=True, shuffle=False, sampler=train_sampler, drop_last=True)
    
    val_dataloader = DataLoader(data, batch_size=batch_size, num_workers=4,
                            pin_memory=True, shuffle=False, sampler=val_sampler, drop_last=True) if len(val_idxs) > 0 else None

    return train_dataloader, val_dataloader