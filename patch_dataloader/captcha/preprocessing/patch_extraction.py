import os
import cv2
import numpy as np
from tqdm import tqdm

try:
    from patch_dataloader.captcha.active_learning.helper_OOP.dataset import ActiveLearningSet
    from patch_dataloader.captcha.active_learning.helper_OOP.volume import NiftiVolume
    from patch_dataloader.captcha.utils.helper import make_dir, patch_per_patient_wnet, print_pairwise_lists


    from patch_dataloader.captcha.utils.utils import generate_and_save_sequential_patches
    from patch_dataloader.captcha.utils.helper import extract_roi
except:
    from captcha.active_learning.helper_OOP.dataset import ActiveLearningSet
    from captcha.active_learning.helper_OOP.volume import NiftiVolume
    from captcha.utils.helper import make_dir, patch_per_patient_wnet, print_pairwise_lists

    from captcha.utils.utils import generate_and_save_sequential_patches
    from captcha.utils.helper import extract_roi
    
def main(skull_stripping_dir, patch_extraction_dir, rough_vessel_dir, AL_train_samples=[], highest_entropies_and_variances_filepath=None, current_ids=[], patch_sizes=96 ):
    
    patch_sizes = patch_sizes if isinstance(patch_sizes, list) else [patch_sizes]
    print("Extracting patches with sizes: ", patch_sizes)
    
    nr_patches = patch_per_patient_wnet # - number of patches we want to extract from one patient
    nr_vessel_patches = nr_patches // 2  # - patches that are extracted around vessels
    nr_empty_patches = nr_patches - nr_vessel_patches  # - random patches without a vessel in the center
    

    patch_extraction_dir_true = [item for item in patch_extraction_dir if 'seg_true_patch_extraction' in item][0]
    patch_extraction_dir = [item for item in patch_extraction_dir if 'weak_patch_extraction' in item][0]
    
    
    print(f"patch_extraction_dir for TRUE seg: {patch_extraction_dir_true}")
    print(f"patch_extraction_dir for WEAK seg: {patch_extraction_dir}")
        
    make_dir(patch_extraction_dir)
    make_dir(patch_extraction_dir_true)
    
    
    print(f"Extracting patches (WNet Dataset) from images in {skull_stripping_dir}")
    
    
    extracted_seg = ActiveLearningSet(patch_extraction_dir)
    already_extracted_seg_imgs = extracted_seg.get_volume_paths()
    
    extracted_seg_true = ActiveLearningSet(patch_extraction_dir_true)
    already_extracted_seg_true_imgs = extracted_seg_true.get_volume_paths()
    
    print(f"Found {extracted_seg.get_len()} already extracted seg files in {patch_extraction_dir}")
    print(f"Found {extracted_seg_true.get_len()} already extracted seg files in {patch_extraction_dir_true}")
    
    
    # List filenames of data after the skull stripping process
    skull_stripped_dataset = ActiveLearningSet(skull_stripping_dir)
    input_list = skull_stripped_dataset.get_volume_paths()
    mask_list = skull_stripped_dataset.get_brain_paths()
    
    true_vessel_dir = [item for item in rough_vessel_dir if 'true_mask' in item][0]
    rough_vessel_dir = [item for item in rough_vessel_dir if 'rough_mask' in item][0]
    
    
    rough_vessel_dataset = ActiveLearningSet(rough_vessel_dir)
    label_list = rough_vessel_dataset.get_label_paths()
    
    true_vessel_dataset = ActiveLearningSet(true_vessel_dir)
    true_label_list = true_vessel_dataset.get_label_paths()
    
    # filter input list to only include the images that are in the current_ids list
    if len(current_ids) > 0:
        input_list = [item for item in input_list if item.split(os.sep)[-1].split('_')[0] in current_ids]
        mask_list = [item for item in mask_list if item.split(os.sep)[-1].split('_')[0] in current_ids]
        label_list = [item for item in label_list if item.split(os.sep)[-1].split('_')[0] in current_ids]
        true_label_list = [item for item in true_label_list if item.split(os.sep)[-1].split('_')[0] in current_ids]
    
    assert len(input_list) == len(mask_list) == len(label_list) == len(true_label_list), f"Number of images ({len(input_list)}), masks ({len(mask_list)}), rough labels ({len(label_list)}) and true labels ({len(true_label_list)}) must be the same"
    
    #print_pairwise_lists(input_list, mask_list, label_list, true_label_list)
    
    ######################à TODO ################################ ACTIVE LEARNING SELECTION !!!!!!!!!!!!
    
    AL_train_samples_id = []
    # if len(AL_train_samples) > 0: 
    #     assert()
    #     AL_train_samples_id = [item.split(os.sep)[-1].split('_')[0] for item in AL_train_samples] # get the patient ID
        
    #     print(f"Found {len(AL_train_samples)} AL train samples")
    #     print(f"We will proceed by extracting sequential patches for each of them")
        
        
    #     for AL_img in AL_train_samples:
            
    #         AL_id = AL_img.split(os.sep)[-1].split('_')[0]
            
    #         AL_save_dir = os.path.join(patch_extraction_dir,AL_id + '_' + str(patch_sizes[0]) + '_al_img.npy')
            
    #         #print(f" Os.listdir(patch_extraction_dir): {os.listdir(patch_extraction_dir)}")
            
    #         if AL_save_dir in os.listdir(patch_extraction_dir):
    #             print(f"Sequential patches for {AL_img} already extracted. Skipping...")
                
    #         else:
    #             generate_and_save_sequential_patches(AL_img, patch_extraction_dir, patch_sizes, highest_entropies_and_variances_filepath)
    
    ######################à TODO ################################ 
    
    # load image, mask and label stacks as matrices	
    for i,img_path in enumerate(tqdm(input_list, desc='Extracting patches for all the patients')):
        
        current_volume = NiftiVolume(img_path, mask_list[i], label_list[i], true_label_list[i])
        current_id = current_volume.get_id()
        
        filter_roi = False
        ROI_mask = False
        
        filter_roi = False if ROI_mask else filter_roi
        assert not (filter_roi and ROI_mask), "filter_roi must be False if padding is True"
                
        if filter_roi or ROI_mask:
            print(f"Filtering ROI for image {current_id}...")
            x_ROI_min, x_ROI_max, y_ROI_min, y_ROI_max, z_ROI_min, z_ROI_max = extract_roi(id=current_id, modality="ct" if "CT" in img_path else "mr")
            print(f"Extracted ROI coordinates: x: {x_ROI_min} - {x_ROI_max} // y: {y_ROI_min} - {y_ROI_max} // z: {z_ROI_min} - {z_ROI_max}")
        
        if current_id in AL_train_samples_id:
            print(f"Patient {current_id} is an AL train sample. Skipping...")
            continue
        
        if len(current_ids)>0 and current_id not in current_ids:
            print(f"Patient {current_id} is not in among the training samples. Skipping...")
            continue 
        
        if current_id in extracted_seg.get_ids():
            print(f"Segmentation Patches of Patient {current_id} already extracted. Skipping...")
            continue
        
        if current_id in extracted_seg_true.get_ids():
            print(f"True Segmentation Patches of Patient {current_id} already extracted. Skipping...")
            continue
        
        current_volume.load_data()
        
        if ROI_mask:
            print(f"Applying ROI mask to image {current_id}...")
            current_volume.mask_volumes_roi(x_ROI_min, x_ROI_max, y_ROI_min, y_ROI_max, z_ROI_min, z_ROI_max)
            # Put 0 outside the ROI and leave the rest unchanged
        
        current_img, current_mask, current_label, current_true_label = current_volume.get_data()
        
        current_nr_extracted_patches = 0  # counts already extracted patches
        
        # make lists in dictionaries for each extracted patch size
        img_patches = {}  # dictionary to save image patches
        label_patches = {}  # dictionary to save label patches
        true_label_patches = {}  # dictionary to save label patches
        
        empty_patches = {}  # dictionary to save empty patches
        empty_labels = {}  # dictionary to save empty patches
        
        for size in patch_sizes:
            img_patches[str(size)] = []
            label_patches[str(size)] = []
            true_label_patches[str(size)] = []
            
            empty_patches[str(size)] = []
            empty_labels[str(size)] = []

        # variables with sizes and ranges for searchable areas
        max_patch_size = max(patch_sizes)
        half_max_size = max_patch_size // 2
        #print(f'{max_patch_size = } --> {half_max_size = }',)
        
        min_row = min_col = half_max_size
        
        
        max_row = current_true_label.shape[0] - half_max_size
        max_col = current_true_label.shape[1] - half_max_size
        
        # ------------------------ ROI -------------------------------
        if filter_roi:
            min_row = half_max_size + x_ROI_min ###DANI
            min_col = half_max_size + y_ROI_min ###DANI
            
            
            max_row = x_ROI_max - half_max_size ###DANI
            max_col = y_ROI_max - half_max_size ###DANI
            
            print(f'{min_row = } --/-- {max_row = }', )
            print(f'{min_col = } --/-- {max_col = }',)
            
            min_depth = z_ROI_min
            max_depth = z_ROI_max
            
            print(f'{min_depth = } --/-- {max_depth = }',)
            
            assert min_row < max_row, f"min_row {min_row} must be smaller than max_row {max_row}"
            assert min_col < max_col, f"min_col {min_col} must be smaller than max_col {max_col}"
            assert min_depth < max_depth, f"min_depth {min_depth} must be smaller than max_depth {max_depth}"
        # -----------------------------------------------------------
        
        
        # -----------------------------------------------------------
        # 1 - EXTRACT RANDOM PATCHES WITH VESSELS IN THE CENTER OF EACH PATCH
        # -----------------------------------------------------------
        # cut off half of the biggest patch on the edges to create the searchable area -> 
        # to ensure that there will be enough space for getting the patch

        if filter_roi:
            searchable_label_area = current_true_label[min_row: max_row, min_col: max_col, min_depth: max_depth]
        else:
            searchable_label_area = current_true_label[min_row: max_row, min_col: max_col, :]
        
            
        vessel_inds = np.asarray(np.where(searchable_label_area == 1)) # find all vessel voxel indices in searchable area
        
        # keep extracting patches while the desired number of patches has not been reached yet, this just in case some
        # patches would be skipped, because they were already extracted (we do not want to have same patches in the set
        # more than once)
        
        while current_nr_extracted_patches < nr_vessel_patches * len(patch_sizes):
            # find given number of random vessel indices (columns)
            random_vessel_inds = vessel_inds[:,np.random.choice(vessel_inds.shape[1], nr_vessel_patches, replace=False)] 
            #print("Random vessel inds shape: ",random_vessel_inds.shape)
            
            for i in tqdm(range(nr_vessel_patches), leave=False, desc='Extracting patches with vessels'):
                
                # stop extracting if the desired number of patches has been reached
                if current_nr_extracted_patches == nr_vessel_patches * len(patch_sizes):
                    break

                # get the coordinates of the random vessel around which the patch will be extracted
                if filter_roi:
                    x = random_vessel_inds[0][i] + half_max_size + x_ROI_min ###DANI
                    y = random_vessel_inds[1][i] + half_max_size + y_ROI_min ###DANI
                    z = random_vessel_inds[2][i] + z_ROI_min ###DANI
                else:
                    x = random_vessel_inds[0][i] + half_max_size
                    y = random_vessel_inds[1][i] + half_max_size
                    z = random_vessel_inds[2][i]
                

                # extract patches of different quadratic sizes with the random vessel voxel in the center of each patch
                for size in patch_sizes:
                    half_size = size // 2
                    random_img_patch = current_img[x - half_size:x + half_size, y - half_size:y + half_size, z]
                    random_label_patch = current_label[x - half_size:x + half_size, y - half_size:y + half_size, z]
                    random_true_label_patch = current_true_label[x - half_size:x + half_size, y - half_size:y + half_size, z]

                    #if not (np.unique(random_label_patch) == np.unique(random_true_label_patch)).all():
                    #    #print(f"Skipping patch because rough and true label have not the same unique values (VESSEL PATCH) [{np.unique(random_label_patch)} != {np.unique(random_true_label_patch)}]")
                    #    #assert(), f"Label and true label must be the same: {np.unique(random_label_patch)} != {np.unique(random_true_label_patch)}"
                    #    continue
                    
                    # just sanity check if the patch is already in the list
                    if any((random_img_patch == x).all() for x in img_patches[str(size)]):
                        #print(f'Skip patch because already extracted. size: {size}')
                        break
                    elif random_true_label_patch.sum() == 0:
                        #print(f'Skip patch because label is empty. size: {size}')
                        empty_patches[str(size)].append(random_img_patch)
                        empty_labels[str(size)].append(random_true_label_patch)
                        assert(), f"Label must not be empty !!!!"
                    else:
                        # append the extracted patches to the dictionaries
                        img_patches[str(size)].append(random_img_patch)
                        label_patches[str(size)].append(random_label_patch)
                        true_label_patches[str(size)].append(random_true_label_patch)
                        current_nr_extracted_patches += 1
                        

        # -----------------------------------------------------------
        # 2 - EXTRACT RANDOM EMPTY PATCHES
        # -----------------------------------------------------------
        # cut off half of the biggest patch on the edges to create the searchable area ->
        # to ensure that there will be enough space for getting the patch
        if filter_roi:
            searchable_mask_area = current_mask[min_row: max_row, min_col: max_col, min_depth: max_depth]
        else:
            searchable_mask_area = current_mask[half_max_size: max_row, half_max_size: max_col, :]
        
        # find all brain voxel indices
        brain_inds = np.asarray(np.where(searchable_mask_area == 1))

        # keep extracting patches while the desired number of patches has not been reached yet, this just in case some
        # patches would be skipped, because they were already extracted (we do not want to have same patches in the set
        # more than once)
        
        while current_nr_extracted_patches < nr_patches * len(patch_sizes):
            
            # find given number of random indices in the brain area
            random_brain_inds = brain_inds[:, np.random.choice(brain_inds.shape[1], nr_empty_patches, replace=False)]
            
            for i in tqdm(range(nr_empty_patches), leave=False, desc='Extracting empty patches'):
                
                # stop extracting if the desired number of patches has been reached
                if current_nr_extracted_patches == nr_patches * len(patch_sizes):
                    break

                # get the coordinates of the random brain voxel around which the patch will be extracted
                if filter_roi:
                    x = random_brain_inds[0][i] + half_max_size + x_ROI_min ###DANI
                    y = random_brain_inds[1][i] + half_max_size + y_ROI_min ###DANI
                    z = random_brain_inds[2][i] + z_ROI_min ###DANI
                else:
                    x = random_brain_inds[0][i] + half_max_size
                    y = random_brain_inds[1][i] + half_max_size
                    z = random_brain_inds[2][i]
                

                # extract patches of different quadratic sizes with the random brain voxel in the center of each patch
                for size in patch_sizes:
                    half_size = size // 2
                    random_img_patch = current_img[x - half_size:x + half_size, y - half_size:y + half_size, z]
                    random_label_patch = current_label[x - half_size:x + half_size, y - half_size:y + half_size, z]
                    random_true_label_patch = current_true_label[x - half_size:x + half_size, y - half_size:y + half_size, z]

                    #if not (np.unique(random_label_patch) == np.unique(random_true_label_patch)).all():
                        #print("Skipping patch because rough and true label are not the same (EMPTY PATCH)")
                        #assert(), f"Label and true label must be the same: {np.unique(random_label_patch)} != {np.unique(random_true_label_patch)}"
                    #    continue
                    
                    # just sanity check if the patch is already in the list
                    if any((random_img_patch == x).all() for x in img_patches[str(size)]):
                        #print(f'Skip patch because already extracted')
                        break
                    elif random_img_patch.sum() == 0:
                         #print(f'Skip patch because empty')
                         break
                    elif random_true_label_patch.sum() == 0:
                        if len(empty_patches[str(size)]) < nr_empty_patches:
                            #print(f'Skip patch because label is empty.')
                            empty_patches[str(size)].append(random_img_patch)
                            empty_labels[str(size)].append(random_true_label_patch)
                        break
                    else:
                        # append the extracted patches to the dictionaries
                        img_patches[str(size)].append(random_img_patch)
                        label_patches[str(size)].append(random_label_patch)
                        true_label_patches[str(size)].append(random_true_label_patch)
                        
                        current_nr_extracted_patches += 1

        assert current_nr_extracted_patches == nr_patches * len(patch_sizes), \
            f"Number of extracted patches: {str(current_nr_extracted_patches)} but should be {str(nr_patches) * len(patch_sizes)}"

        #print(f'Saving extracted patches...')
        
        # save extracted patches as numpy arrays
        for size in patch_sizes:
            assert len(img_patches[str(size)]) == len(true_label_patches[str(size)]), "Number of image patches and label patches are not equal"
            #print(f'Number of extracted patches: {len(img_patches[str(size)])} ({nr_vessel_patches} vessel + {nr_empty_patches} empty)')
            
            directory = patch_extraction_dir
            directory_true = patch_extraction_dir_true
            
            save_extracted_patches(current_id, img_patches, label_patches, size, directory)
            save_extracted_patches(current_id, img_patches, true_label_patches, size, directory_true, save_pngs=True)
            save_extracted_patches(current_id, empty_patches, empty_labels ,size, directory_true.replace("true", "empty"))
        
        #print(f'Patches extracted and saved successfully')
        
        
    # End of for loop over all images    
    print('All patches extracted and saved successfully')

def save_extracted_patches(current_id, img_patches, label_patches, size, directory, png_dir='png_final', save_pngs=False):
    make_dir(directory)
    img_saving_path = os.path.join(directory,current_id + '_' + str(size) + '_img.npy')
    np_img_patches = np.asarray(img_patches[str(size)])
    np.save(img_saving_path, np_img_patches)
    #print(f'Image patches saved to {img_saving_path}')
            
    label_saving_path = os.path.join(directory,current_id + '_' + str(size) + '_label.npy')
    np_label_patches = np.asarray(label_patches[str(size)])
    np.save(label_saving_path, np_label_patches)
    #print(f'Label patches saved to {label_saving_path}')
    
    if 'empty_patch' in directory:
        print(f"Number of empty patches: {len(img_patches[str(size)])}!!!")

    if save_pngs:
        png_directory = os.path.dirname(os.path.dirname(directory.rstrip("/")))
        png_directory = os.path.join(png_directory, png_dir)
        make_dir(png_directory)
        print(f"Saving patches as PNG to {png_directory}")
        
        for i,label_p in enumerate(np_label_patches):
            save_png(label_p, os.path.join(png_directory, f"{current_id}_{size}_label_{i}.png"))

def save_png(img_array,title): 
    _, max_val = np.min(img_array), np.max(img_array)
    # Normalize the image between 0 and 255
    img_array = (img_array) * (255 / (max_val))
    cv2.imwrite(title, img_array)
    assert(max_val > 0), f"Max value of the image is {max_val} for {title} {np.unique(img_array)}"