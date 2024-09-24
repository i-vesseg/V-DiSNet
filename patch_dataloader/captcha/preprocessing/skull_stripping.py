"""
This file removes the skull from the MRI images.

Takes as input three .nii files : the original image, the mask, and the vessel_mask (which serves as the vessel labels).
Outputs three .nii files : _img, _mask, _label with the skull removed.
"""
import os
try:
    from patch_dataloader.captcha.active_learning.helper_OOP.dataset import ActiveLearningSet, TrainingSet, TestSet
    from patch_dataloader.captcha.active_learning.helper_OOP.volume import NiftiVolume
    from patch_dataloader.captcha.utils.helper import make_dir, print_pairwise_lists
except:
    from captcha.active_learning.helper_OOP.dataset import ActiveLearningSet, TrainingSet, TestSet
    from captcha.active_learning.helper_OOP.volume import NiftiVolume
    from captcha.utils.helper import make_dir, print_pairwise_lists
from tqdm import tqdm

def main(current_dataset, target_dir, standardize_volume):
    
    make_dir(target_dir)
    assert isinstance(current_dataset, ActiveLearningSet) or isinstance(current_dataset, TestSet), f"Invalid training set type: {type(current_dataset)}"
    
    if standardize_volume:
        print("After skull stripping, each volume will be standardized by using its own mean and standard deviation.")
        
    
    already_skull_stripped = TrainingSet(target_dir)
    already_skull_stripped.get_volume_paths()
    
    print(f"Found {already_skull_stripped.get_len()} already skull-stripped files in {target_dir}")
    
    
    ## input_list is the full images in .nii format
    input_list = current_dataset.get_volume_paths()
    mask_list = current_dataset.get_brain_paths()
    label_list = current_dataset.get_label_paths()
        
    #print_pairwise_lists(input_list, mask_list, label_list)
        
    assert len(input_list) == len(mask_list) == len(label_list), (len(input_list),len(mask_list), len(label_list))
    assert len(input_list) >0, f"No files found in the input directory {current_dataset.get_datapath()}"
    
    # load image, mask and label stacks as matrices	
    for i,img_path in enumerate(tqdm(input_list)):
                
        print(f"Processing {img_path}...")
        
        
        current_volume = NiftiVolume(img_path, mask_list[i], label_list[i])
        current_id = current_volume.get_id()
        
        if current_id not in current_dataset.get_ids():
            print(f"Patient {current_id} is not in among the training samples. Skipping...")
            continue 
        
        
        # skip if already skull-stripped
        if current_id in already_skull_stripped.get_ids():
            print(f"{current_id} already skull-stripped. Skipping...")
            
            continue
        
        current_volume.load_data()
        print(f"MIN VALUE: {current_volume.get_min()} --- MAX VALUE: {current_volume.get_max()}")
        current_volume.skull_strip(bg_value=current_volume.get_min())
        
        # Standardize the image
        if standardize_volume:
            print('Standardizing image...')
            current_volume.standardize()
        
        # save to new file as masked version of original data 
        print('-----------------')
        current_volume.save_volume(os.path.join(target_dir,current_id + '_img.nii.gz'))
        current_volume.save_brain_volume(os.path.join(target_dir,current_id + '_mask.nii.gz'))
        current_volume.save_vessel_volume(os.path.join(target_dir,current_id + '_label.nii.gz'))
        
        print(f'Image, mask and label saved to: {target_dir}')
        print('-----------------')
        
    print('Skull stripping completed')