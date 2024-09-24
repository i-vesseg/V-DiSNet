import os, shutil

from tqdm import tqdm
try:
    from patch_dataloader.captcha.active_learning.helper_OOP.dataset import ActiveLearningSet
    from patch_dataloader.captcha.active_learning.helper_OOP.volume import NiftiVolume
    from patch_dataloader.captcha.utils.helper import make_dir, print_pairwise_lists
except:
    from captcha.active_learning.helper_OOP.dataset import ActiveLearningSet
    from captcha.active_learning.helper_OOP.volume import NiftiVolume
    from captcha.utils.helper import make_dir, print_pairwise_lists

def main(skull_stripping_dir, grid_dir, patch_annotation_dir, current_ids=[]):
            
    make_dir(patch_annotation_dir)
    
    
    print(f"Simulating patch annotation for images in {skull_stripping_dir}")
    
    
    annotated_dataset = ActiveLearningSet(patch_annotation_dir)
    print(f"Found {annotated_dataset.get_len()} already annotated files in {patch_annotation_dir}")
    
    skull_stripped_dataset = ActiveLearningSet(skull_stripping_dir)
    input_list = skull_stripped_dataset.get_volume_paths()
    mask_list = skull_stripped_dataset.get_brain_paths()
    
    grid_dataset = ActiveLearningSet(grid_dir)
    grid_dataset.get_volume_paths()
    grid_label_list = grid_dataset.get_label_paths()

    assert len(input_list) == len(grid_label_list), (len(input_list), len(grid_label_list))
    assert len(input_list) >0, "No input files found"
    
    #print_pairwise_lists(input_list, mask_list, grid_label_list)
    
    for img, brain_mask, grid_label in tqdm(zip(input_list, mask_list, grid_label_list), total=len(input_list)):
        
        current_volume = NiftiVolume(img, brain_mask, grid_label)
        current_id = current_volume.get_id()
        
        # skip if not in train samples
        if len(current_ids) > 0 and current_id not in current_ids:
            print(f"Patient {current_id} is not in among the training samples. Skipping...")
            continue
        
        if current_id in annotated_dataset.get_ids():
            print(f"Skipping {img.split('/')[-1]}: already annotated")
            
            continue
        
        print('Start annotating patch...')
        print('-----------------')
        
        
        shutil.copy(img,os.path.join(patch_annotation_dir,os.path.basename(img)))
        shutil.copy(brain_mask,os.path.join(patch_annotation_dir,os.path.basename(brain_mask)))
        shutil.copy(grid_label,os.path.join(patch_annotation_dir,os.path.basename(grid_label)))
        print(f'Patch annotated')
        
    print('-'*50)
    print('All patches annotated')