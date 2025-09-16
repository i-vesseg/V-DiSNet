# Import the necessary packages
import os
import torch

print(torch.__version__) # 1.11.0+cu112
print(torch.cuda.is_available()) # True

print(f"Current working directory: {os.getcwd()}")

try:
    from patch_dataloader.captcha.active_learning.helper_OOP.dataset import ActiveLearningSet
    from patch_dataloader.captcha.utils.helper import set_seed
    from patch_dataloader.captcha.preprocessing.skull_stripping import main as skull_stripping
    from patch_dataloader.captcha.preprocessing.grid import main as grid_generation
    from patch_dataloader.captcha.preprocessing.simulate_patch_annotation import main as simulate_patch_annotation
    from patch_dataloader.captcha.preprocessing.rough_mask import copy_list_to_dir ,main as rough_mask_generation
    from patch_dataloader.captcha.preprocessing.patch_extraction import main as seg_patch_extraction
    from patch_dataloader.captcha.patch_dataloader import get_dataloader
except:
    from captcha.active_learning.helper_OOP.dataset import ActiveLearningSet
    from captcha.utils.helper import set_seed
    from captcha.preprocessing.skull_stripping import main as skull_stripping
    from captcha.preprocessing.grid import main as grid_generation
    from captcha.preprocessing.simulate_patch_annotation import main as simulate_patch_annotation
    from captcha.preprocessing.rough_mask import copy_list_to_dir ,main as rough_mask_generation
    from captcha.preprocessing.patch_extraction import main as seg_patch_extraction
    from captcha.patch_dataloader import get_dataloader


def main(preprocessed_data_dir, standardize_volume, use_true_labels_train, original_data_dir, patch_size,
         clustering_method="kmeans", apply_dilation=True, **dataloader_parameters):
    
    """
    Main function to preprocess the data and get the dataloader for Vessel Patches
    
    Args:
    - preprocessed_data_dir: str, path to the preprocessed data (output dir)
    - standardize_volume: bool, if True, the volume will be standardized by using its own mean and standard deviation
    - use_true_labels_train: bool, if True, the true labels are used to extract patches (if False, the rough mask is used)
    - original_data_dir: str, input dir for the original data
    - patch_size: list, size of the patches
    
    - clustering_method: str, method for the clustering (kmeans or gmm)
    - apply_dilation: bool, if True, the dilation is applied to the rough mask
    """
    # Set the seed for reproducibility 
    def print_parameters():
        print(f"INPUT  --- Path to the data: {original_data_dir}")
        print(f"OUTPUT --- Path to the preprocessed data: {preprocessed_data_dir}\n")
        
        print(f"Current seed: {seed}")
        print(f"Standardize volume: {standardize_volume}")
        print(f"Use true labels for training: {use_true_labels_train}")
        print(f"Clustering method: {clustering_method}")
        print(f"Apply dilation: {apply_dilation}")
        print(f"Patch size: {patch_size}")

    seed = 42
    set_seed(seed)
    print_parameters()
    # -------------------------------- Training sample list -------------------------------- #
    current_training_set = ActiveLearningSet(original_data_dir)
    current_training_set.get_volume_paths()
    current_ids = current_training_set.get_ids()
    print(f"Current ids: {current_ids} (Length: {len(current_ids)}")
    
    print(f"Training set len: {current_training_set.get_len()} Patient")
    
    # --------------------- Step 1.1: Skull stripping --------------------- #
    print("Step 1.1: Skull stripping")  
    
    #Output dir:
    skull_stripping_train_dir=os.path.join(preprocessed_data_dir,"1_skull_stripped_images/") #Dir containing the skull-stripped images
    
    skull_stripping(current_training_set, skull_stripping_train_dir, standardize_volume)
    current_skull_stripped_train_set = ActiveLearningSet(skull_stripping_train_dir)
    # --------------------- Step 1.2: Grid generation --------------------- #
    print("Step 1.2: Grid generation")

    #Output dir:
    grid_dir=os.path.join(preprocessed_data_dir,"2_grid_images/")
    grid_generation(current_skull_stripped_train_set, grid_dir, patch_size, current_ids) 
    # ---------- Step 1.2.2: Simulate patches annotation ---------- #
    print("Step 1.2.2: Simulate patches annotation process")

    #Output dir:
    patch_annotation_dir=os.path.join(preprocessed_data_dir,"2_annotated_patches/") #Dir containing the simulated patches annotation 
    simulate_patch_annotation(skull_stripping_train_dir, grid_dir, patch_annotation_dir, current_ids)

    # --------------------- Step 1.3: Rough mask generation --------------------- #
    print("Step 1.3: Rough and True vessel patches generation")
        
    # True mask extraction
    true_mask_dir=os.path.join(preprocessed_data_dir,"3_true_mask_dir/") # Dir in which have to be saved the true labels images
    copy_list_to_dir(current_skull_stripped_train_set.get_label_paths(), true_mask_dir)
    
    dil_str = "_dil" if apply_dilation else ""
    rough_mask_dir=os.path.join(preprocessed_data_dir,f"3{dil_str}_{clustering_method}_rough_mask_dir/") # Dir in which have to be saved the images after the rough_mask.py process 
    print(f"Current ids for rough mask generation: {current_ids} (Length: {len(current_ids)}")
    rough_mask_generation(patch_annotation_dir, rough_mask_dir, patch_size, clustering_method=clustering_method, apply_dilation=apply_dilation, current_ids=current_ids)
   
    # This dir is the input to the WNET dataset generation (seg_patch_extraction)
    rough_mask_dir = [true_mask_dir, rough_mask_dir]
    print(f"Current rough mask dir: {rough_mask_dir}")
    
    # --------------------- Step 1.5: Patches Dataset generation --------------------- #
    print("Step 1.5: Patches Dataset generation")
        
    #Output dirs:
    patch_extraction_dir=[os.path.join(preprocessed_data_dir,"5_seg_true_patch_extraction/"), os.path.join(preprocessed_data_dir,f"5{dil_str}_{clustering_method}_seg_weak_patch_extraction/")]
    print(f"Current patch extraction dir: {patch_extraction_dir}")
    
    seg_patch_extraction(skull_stripping_train_dir, patch_extraction_dir, rough_mask_dir, current_ids=current_ids, patch_sizes=patch_size)

    patch_extraction_dir = patch_extraction_dir[0] if use_true_labels_train else patch_extraction_dir[1]
    print(f"The parameter use_true_labels_train is set to {use_true_labels_train}. The patch extraction dir will be set to {patch_extraction_dir}")
    
    # -------------------------------- Step 4: Get dataloader -------------------------------- #
    print("Get dataloader for training and validation set")
    
    dataloader_metadata_filepath = os.path.join(preprocessed_data_dir, "dataloader_metadata")
    train_dataloader, val_dataloader = get_dataloader(patch_extraction_dir, dataloader_metadata_filepath, current_ids=current_ids, **dataloader_parameters)

    return train_dataloader, val_dataloader
                
if __name__ == "__main__":
    
    # add argument
    import argparse, time

    parser = argparse.ArgumentParser(description='Dataset')
    parser.add_argument('--dataset', type=str, help='Dataset Name')

    args = parser.parse_args()
    
    print(f"------------------- Dataset: {args.dataset} -------------------")
    time.sleep(3)
    
    # ----------------------------------- #
    # original_data_dir = INPUT_DIR/train
    # preprocessed_data_dir = OUTPUT_DIR
    # ----------------------------------- #
    
    if args.dataset == "IXI":
        # IXI dataset (for Dictionary Learning and Contrastive Learning)
        original_data_dir = "/data/falcetta/brain_data/IXI/processed/numpy_renamed/train"
        preprocessed_data_dir = "/data/falcetta/brain_data/IXI/processed/numpy_renamed/patches_preprocessed"
        preprocessed_data_dir = "/data/falcetta/brain_data/IXI/processed/numpy_renamed/patches_preprocessed_with_empty"
    elif args.dataset == "CAS":
        # CAS dataset (for WNET Training)
        original_data_dir = "/data/falcetta/brain_data/CAS/preprocessed/numpy/train"
        preprocessed_data_dir = "/data/falcetta/brain_data/CAS/preprocessed/numpy/patches_preprocessed"
        preprocessed_data_dir = "/data/falcetta/brain_data/CAS/preprocessed/numpy/patches_preprocessed_with_empty"
    elif args.dataset == "OASIS":
        # OASIS dataset (for WNET Training)
        original_data_dir = "/data/falcetta/brain_data/OASIS/processed/numpy/train"
        preprocessed_data_dir = "/data/falcetta/brain_data/OASIS/processed/numpy/patches_preprocessed"
    elif args.dataset == 'CASJ':
        original_data_dir = "/data/falcetta/brain_data/CASJ/preprocessed/numpy/train"
        preprocessed_data_dir = "/data/falcetta/brain_data/CASJ/preprocessed/patches_preprocessed_with_empty"
    elif args.dataset == 'IXIJ':
        original_data_dir = "/data/falcetta/brain_data/IXIJ/processed/numpy_renamed/train"
        preprocessed_data_dir = "/data/falcetta/brain_data/IXIJ/processed/patches_preprocessed_with_empty"
        
    standardize_volume = True
    use_true_labels_train = True
    patch_size = 32
    
    dataloader_parameters = {
        "patch_size": patch_size,
        "validation_split": 0.2,
        "normalize": True,
        "batch_size": 200,
        "augmentation": False,
        "pixel_wise": True,
        "load_empty": True,
    }
    main(preprocessed_data_dir, standardize_volume, use_true_labels_train, original_data_dir, **dataloader_parameters)