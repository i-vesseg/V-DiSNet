import os
from tqdm import tqdm

try:
    from patch_dataloader.captcha.active_learning.helper_OOP.slice import MedicalImageSlice
    from patch_dataloader.captcha.active_learning.helper_OOP.volume import NiftiVolume

    from patch_dataloader.captcha.utils.helper import make_dir, print_pairwise_lists
    from patch_dataloader.captcha.active_learning.helper_OOP.dataset import ActiveLearningSet, GridDataset, TestSet
except:
    from captcha.active_learning.helper_OOP.slice import MedicalImageSlice
    from captcha.active_learning.helper_OOP.volume import NiftiVolume

    from captcha.utils.helper import make_dir, print_pairwise_lists
    from captcha.active_learning.helper_OOP.dataset import ActiveLearningSet, GridDataset, TestSet




def main(current_skull_stripped_train_set, grid_dir, patch_size, current_ids=[]):
    
    assert isinstance(current_skull_stripped_train_set, ActiveLearningSet) or isinstance(current_skull_stripped_train_set, TestSet), f"Invalid training set type: {type(current_skull_stripped_train_set)}"
        
    make_dir(grid_dir)

    
    
    print(f"Creating grid for images in {current_skull_stripped_train_set.get_datapath()}")
    
    
    grid_dataset = GridDataset(grid_dir)
    grid_dataset.get_grid_volume_paths()
    
    print(f"Found {grid_dataset.get_len()} already grid files in {grid_dir}")
    
    
    input_list = current_skull_stripped_train_set.get_volume_paths()
    mask_list = current_skull_stripped_train_set.get_brain_paths()
    vessel_list = current_skull_stripped_train_set.get_label_paths()
    
    #print_pairwise_lists(input_list, mask_list)
    
    print("Creating grid...")
    
    # load image, mask and label stacks as matrices
    for i, img_path in enumerate(tqdm(input_list)):
        
        current_volume = NiftiVolume(img_path, mask_list[i], vessel_list[i])
        current_id = current_volume.get_id()
         
        if len(current_ids)>0 and current_id not in current_ids:
            print(f"Patient {current_id} is not in among the training samples. Skipping...")
            continue 
        
        if current_id in grid_dataset.get_ids():
            print(f"Already grid {current_id + '_vessel_grid.nii.gz'}. Skip this image.")
            
            continue
        
        current_volume.load_data()
        
        print(f"Processing {img_path}")
        
                
        current_volume.create_empty_grid()
        z, z_slices = current_volume.get_brain_z_slices_idx()
        
        x_dim, y_dim, z_dim = current_volume.get_shape()
        
        # Check that patches size fits in the image
        if x_dim % patch_size == 0 and y_dim % patch_size == 0:
            print(f"Patch size {patch_size} does not perfectly fit in image of size {x_dim}x{y_dim}.")
            
            
        # proceed slice by slice (l = slice along the z axis)
        for slice_id in z_slices:
            
            #print(f'Slice: #{l}')
            
            slice_l, label_l, brain_l = current_volume.get_slice(axis=2, index=slice_id, get_brain=True)
            current_slice = MedicalImageSlice(slice_l, slice_id ,vessel_data=label_l, brain_data=brain_l)
            _, vessel_patches, _,  patch_positions = current_slice.get_consecutive_patches(patch_size=patch_size, overlap_x=0.0, overlap_y=0.0, get_mask_patches=True)
            
            current_slice.create_grid(vessel_patches, patch_positions)
            current_volume.reconstruct_grid(current_slice.get_grid(), slice_id)
                    
        
        
        # SAVE AS NIFTI
        print('Grid generation finished. Saving grid...')
        grid_filepath = os.path.join(grid_dir, current_id + '_vessel_grid.nii.gz')
        current_volume.save_grid(grid_filepath)
        
        print(f'Saved grid in {grid_filepath}')
        print('-----------------')
        
    print('Grid generation finished.')