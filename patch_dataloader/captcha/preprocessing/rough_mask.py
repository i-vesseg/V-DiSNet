import shutil, os
import numpy as np

from tqdm import tqdm

try:
    from patch_dataloader.captcha.active_learning.helper_OOP.dataset import ActiveLearningSet
    from patch_dataloader.captcha.active_learning.helper_OOP.slice import MedicalImageSlice
    from patch_dataloader.captcha.active_learning.helper_OOP.volume import NiftiVolume
    from patch_dataloader.captcha.utils.helper import make_dir, print_pairwise_lists
    from patch_dataloader.captcha.utils.helper import create_and_save_nifti
except:
    from captcha.active_learning.helper_OOP.dataset import ActiveLearningSet
    from captcha.active_learning.helper_OOP.slice import MedicalImageSlice
    from captcha.active_learning.helper_OOP.volume import NiftiVolume
    from captcha.utils.helper import make_dir, print_pairwise_lists
    from captcha.utils.helper import create_and_save_nifti

def copy_list_to_dir(label_list, dest_dir):
    
    make_dir(dest_dir)            
    print(f"Copying true labels to {dest_dir}")
    
            
    for item in label_list:
        item_new_filepath = os.path.join(dest_dir, os.path.basename(item))
                
        if os.path.isfile(item_new_filepath):
            print(f"File {os.path.basename(item)} already exists in {dest_dir}")
        else:
            print(f"Copying {os.path.basename(item)}... ")
            shutil.copy(item, item_new_filepath)
                
        
        
def main(patch_annotation_dir, rough_mask_dir, patch_size, clustering_method='kmeans', apply_dilation =True, current_ids=[]):
     
    make_dir(rough_mask_dir)
    
    
    print(f"Generating rough mask for images in {patch_annotation_dir}")
    
    print(f"Clustering method: {clustering_method}")
    
    
    if apply_dilation:
        print("Applying dilation to the rough mask")
        

    rough_mask_dataset = ActiveLearningSet(rough_mask_dir)
    rough_mask_list = rough_mask_dataset.get_label_paths(filtering='_label_rough')
    
    print(f"Found {len(rough_mask_list)} already rough mask files in {rough_mask_dir}")
    
    # List filenames of data after the skull stripping process
    annotated_dataset = ActiveLearningSet(patch_annotation_dir)
    input_list = annotated_dataset.get_volume_paths()
    mask_list = annotated_dataset.get_brain_paths()
    label_list = annotated_dataset.get_label_paths(filtering='vessel_grid')
    
    #print_pairwise_lists(input_list, mask_list, label_list)
    # load image, mask and label stacks as matrices
    for i, img_path in enumerate(tqdm(input_list)):
        
        print(f'Processing {img_path}...')
        
        
        current_volume = NiftiVolume(img_path, mask_list[i], label_list[i])
        current_id = current_volume.get_id()
        
        if len(current_ids)>0 and current_id not in current_ids:
            print(f"Patient {current_id} is not in among the training samples. Skipping...")
            continue 
        
        if current_id in rough_mask_dataset.get_ids():
            print(f"Skipping {current_id} because rough mask already exists")
            
            continue
        
        current_volume.load_data()
        #current_volume.standardize()
        #print(f'WARNING !!! Enhancing vessels...')
        #current_volume.enhance_vessels_3d()
        #assert current_volume.get_max() > 10, f"Max value of the volume is {current_volume.get_max()}"
        
        
        current_volume.create_empty_grid(as_type=np.float32)
        z, z_slices = current_volume.get_brain_z_slices_idx()
        
        
        
        for slice_id in tqdm(z_slices, leave=False): # Iterate over z slices
            
            slice_l, label_l, brain_l = current_volume.get_slice(axis=2, index=slice_id, get_brain=True)
            current_slice = MedicalImageSlice(slice_l, slice_id, brain_data=brain_l ,vessel_data=label_l)
            patches_tuple = current_slice.get_consecutive_patches(patch_size=patch_size, overlap_x=0.0, overlap_y=0.0, get_mask_patches=True)
            
            rough_patches = current_slice.generate_rough_prediction(patches_tuple, clustering_method, apply_dilation)
            current_slice.reconstruct_slice(rough_patches, patch_positions=patches_tuple[3])
            current_volume.reconstruct_volume(current_slice.get_reconstructed_slice(), slice_id)
        
            
        
        
        print(f'Rough Label Generated. Saving...')
        create_and_save_nifti(current_volume.get_reconstructed_volume(), os.path.join(rough_mask_dir, current_id + '_label_rough.nii.gz'))
        print(f'Rough Label Saved.')
        print('-------------------')

    print('-'* 50)
    print('Rough Mask Generation Completed.')



    
    