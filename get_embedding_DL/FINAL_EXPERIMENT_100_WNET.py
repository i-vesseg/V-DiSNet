# -------------------------- SAMPLED CAS DATASET --------------------------
# R     ==> RANDOM  +   NO CLASS    (No class)          (RANDOM)
# C     ==> RANDOM  +   STRATIFIED  (class from latent) (RANDOM)
# CF    ==> FPS     +   STRATIFIED  (class from latent) (FPS Distance from 2D)
# CFW   ==> FPS     +   NO CLASS    (No class)          (FPS Distance from 2D)
# CFWX  ==> FPS     +   NO CLASS    (No class)          (FPS Distance from latent)
# CFX   ==> FPS     +   STRATIFIED  (class from latent) (FPS Distance from latent)


# ALL   ==> COMPLESE CAS DATASET
# IXI   ==> IXI DATASET (Used for dict learning)
# SOTA  ==> (CF) DIVERSITY SAMPLING OVER SOTA CLASS

# -------------------------- Take Values --------------------------
# 1. 0.1/100 ==> 0.1% of the dataset (0.001)
# 2. 1/100   ==> 1% of the dataset (0.01)
# 5. 5/100   ==> 5% of the dataset (0.05)
# 3. 10/100  ==> 10% of the dataset (0.1)
# 4. 30/100  ==> 30% of the dataset (0.3)
# 5. 50/100  ==> 50% of the dataset (0.5)
# 6. 75/100  ==> 75% of the dataset (0.75)
# 7. 100/100 ==> 100% of the dataset (1.0)

import os
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from utils import mk_dir

from itertools import islice

def chunked_iterable(iterable, size):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])

def chunked_array(array, size):
    array_chunks = [array[i:i + size] for i in range(0, len(array), size)]
    return array_chunks


def main():    
    # -------------------------- START TEST --------------------------
    test_mode = False
    train_set_dir = "/data/falcetta/brain_data/IXIJ/processed/"
    test_set_dir = os.path.join(train_set_dir,"numpy_renamed/test")
    metadata_path = os.path.join(train_set_dir,'patches_preprocessed_with_empty/dataloader_metadata/metadata.npz')
    
    for current_seed in [42, 43, 44, 45, 46, 47, 48, 49, 50]:
        print(f"Current seed: {current_seed}")
        for dataset_name in ['IXI','IXI_weak']: # Sota baselines
            run_final_experiment(test_mode, dataset_name,train_set_dir,test_set_dir, metadata_path, current_seed)
                
def run_final_experiment(test_mode, dataset_name,train_set_dir,test_set_dir, metadata_path, current_seed=42):
    
    # -------------------------- Load Metadata --------------------------
    saved_data_path = os.path.join(train_set_dir,f'vessels_{dataset_name}')

    print("\n### ------------------------------------------------------------------- ###")
    print(f"Running final experiment for Dataset:{dataset_name}")
    print(f"Test mode: {test_mode} with Testset dir: {test_set_dir}")
    print("### ------------------------------------------------------------------- ###\n")
    
    print(f"Loading training patches from {saved_data_path}")
    
    print("Loading metadata from the complete dataset")
    metadata = np.load(metadata_path)
    metadata = metadata['mean'].item(), metadata['std'].item()
    print(f"Metadata {metadata}")
    # ----------------------------------------------------- #
    print("Loading Complete dataset")
    
    X_test_all = np.load(os.path.join(saved_data_path, f'img_list_{dataset_name}_all.npy'))
    X_test_mask = np.load(os.path.join(saved_data_path, f'vess_list_{dataset_name}_all.npy'))

    X_test_all_empty = np.load(os.path.join(saved_data_path, f'empty_img_list_{dataset_name}_all.npy'))
    X_test_mask_empty = np.load(os.path.join(saved_data_path, f'empty_vess_list_{dataset_name}_all.npy'))

    X_test_all = np.concatenate((X_test_all, X_test_all_empty), axis=0)
    X_test_mask = np.concatenate((X_test_mask, X_test_mask_empty), axis=0)

    # ----------------------------------------------------- #
        
    print(f"Loading dataset {dataset_name}")
    
    X, X_mask = X_test_all, X_test_mask
    print(f"Dataset for train: {X.shape}, {X_mask.shape}")
    
    #DATA RETRIVAL
    model_filepath = f'FINAL_EXP_100_WNET/{dataset_name}/seed_{current_seed}'
    mk_dir(model_filepath)
    print(f"Model will be saved at: {model_filepath}")
    
    #Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Training on {device} ")
        
    loss = 'dice'
    ignore_background = True
    alpha = 0.0

    # Instantiate the loss function
    print(f"Instantiating loss ({loss}) function...")
        
    if loss in ['dice_loss', 'DiceLoss', 'dice']:
        criterion = GDiceLoss #Dice
        print(f"Using custom dice loss function ({ignore_background = })...")
    elif loss == 'cldice':
        criterion = soft_dice_cldice(alpha=alpha) # alpha=0 -> dice, alpha=1 -> cldice
    else:
        criterion = torch.nn.BCELoss()
        

    if loss in ['dice_loss', 'DiceLoss', 'dice']:
        print(f"Loss function: {criterion.__name__}")
    else:
        print(f"Loss function: {criterion}")
        
        
    # METRICS
    def get_metric_dice():
        threshold = 0.2 # From paper
        metric_dice = torchmetrics.Dice(average='micro', ignore_index=0, threshold=threshold) # Ignore background, threshold=0.2 (from paper)
        return metric_dice.to(device)

    metric_dice = get_metric_dice()
        

    #TRAIN/VALIDATION SPLIT#    
    X_train, X_val, X_train_mask, X_val_mask = train_test_split(X, X_mask, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, {X_train_mask.shape} - Val: {X_val.shape}, {X_val_mask.shape}")

    print("Loading dataset for Wnet...")
    train_dataset = WNetPatchDataset(images_array=X_train, labels_array=X_train_mask,
                                    transform=transforms.ToTensor(), metadata=metadata)
    val_dataset = WNetPatchDataset(images_array=X_val, labels_array=X_val_mask,
                                transform=transforms.ToTensor(), metadata=metadata)

    #SAMPLER & DATALOADER#
    print(f"Creating dataloaders...")
    train_dataloader = DataLoader(train_dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=128)

    val_dataloader = DataLoader(val_dataset,
                            shuffle=False,
                            num_workers=8,
                            batch_size=256)
    
    import gc
    from torch import nn

    #MODEL, OPTIMIZER#
    n_classes = 1
    in_channels = 1
    patch_size = 32
    lr = 0.0001 

    dropout = 0.1
    bn = False

    activation = nn.ReLU() #nn.ReLU() , nn.LeakyReLU()
    print(f"Instantiating Wnet model with activation {activation}...")
    model = WNetSeg(n_classes=n_classes, patch_size=patch_size, in_channels=in_channels, activation=activation, final_activation=nn.Sigmoid(), dropout=dropout, bn=bn)

    # Move the model to the GPU
    #print(f'Model summary: {model}')
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Instantiating optimizer {opt}...")

    gc.collect()
        
    if not test_mode:
        datetime_path = ''
        best_model_path = os.path.join(model_filepath,datetime_path)

        # Create the summary writer object for the loss and accuracy plots in the train metadata folder
        writer = SummaryWriter(os.path.join(model_filepath,datetime_path,'tensorboard'))
        print(f"Tensorboard logs will be saved in {os.path.join(model_filepath,datetime_path,'tensorboard')}")

        # Create the best_model_path directory if it does not exist
        mk_dir(best_model_path)
        print(f"Best models will be saved in {best_model_path} ")
        
        # Check if best model exist and put test_mode to True
        if os.path.exists(os.path.join(best_model_path, 'best_model.pth')):
            print(f"Best model already exist in {best_model_path}")
            print(f"Test mode activated")
            test_mode = True
           
    if not test_mode:
        epochs = 300
        patience = 50
        max_steps = 100000

        print(f"Step per epoch: {len(train_dataloader)} (Total max_steps: {len(train_dataloader)*epochs})")

        if max_steps is not None:
            print(f"Max allowed steps: {max_steps}")
            
        #TRANING + VALIDATION 
        min_val_loss = np.inf
        best_val_dice = 0.0
        num_epochs_no_improvement = 0 # Early stopping counter
        cumul_step = 0
        last_best_epoch = 0

        evaluation_method = 'loss' # Choose between loss, dice
        assert evaluation_method.lower() in ['loss', 'dice'], f'Evaluation method not supported: choose between loss, dice (current: {evaluation_method})'
        print(f"Training for {epochs} epochs...")
    else:
        epochs = 0
            
    import time

    for epoch in range(epochs):
        print(" ------------------------------------------------------- Training ------------------------------------------------------- ")
        #MODEL FITTING (TRAINING)
        train_loss = 0.0
        cnt_train = 0
        start_time = time.time()

        for i, (samples, labels) in enumerate(train_dataloader):
            
            if max_steps is not None and cumul_step > max_steps:
                print(f"Max steps reached ({max_steps}). Stopping training after the last validation round...")
                break
            
            #Put your input samples and labels in the GPU, if available
            samples, labels = samples.to(device).float(), labels.to(device).float()
            
            assert samples.type() == labels.type(), f"Samples ({samples.shape}) and labels ({labels.shape}) must have the same type ({samples.type()} vs {labels.type()})"
            assert samples.shape[0] == labels.shape[0] and samples.shape[1] == labels.shape[1], f"Samples ({samples.shape}) and labels ({labels.shape}) must have the same number of samples and channels"
            
            # clear gradients
            opt.zero_grad() 
            
            # forward pass
            out = model(samples) 
            
            loss = criterion(out, labels, ignore_background=ignore_background) if ignore_background else criterion(out, labels)
            # backward pass
            loss.backward() 
            
            # update weights/parameters
            opt.step() 

            
            train_dice = metric_dice(out, labels.long()) 
            cumul_train_dice = metric_dice.compute()
        
            # UNCOMMENT TO PRINT TRAINING INFO EVERY 60 SECONDS OR WHEN THE LOSS IS TOO HIGH
            # if loss.item() > 3 or (time.time() - start_time)>60:
            #     cumul_step = (i) + epoch*len(train_dataloader)
            #     print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_dataloader)} - {cumul_step}], Train Loss: {loss.item():.2e}, Step-Time: {(time.time() - start_time):.2f}s")
            #     writer.add_scalar('Training/Loss', loss.item(), cumul_step) # Write loss to TensorBoard
                
            #     print(f"Train Dice score: {100*train_dice:.2f}% (Cumul: {100*cumul_train_dice:.2f}%)")
                
            #     writer.add_scalar('Training/Dice', train_dice, cumul_step)
            #     writer.add_scalar('Training/Cumul_Dice', cumul_train_dice, cumul_step)
                
                
            #     start_time = time.time() 7
            
            # update training loss for each batch
            train_loss += loss.item()
            cnt_train += 1
        
        #End of one epoch
        cumul_step = (i) + epoch*len(train_dataloader)
        # Compute final dice score (average over all batches)
        final_dice_train = metric_dice.compute()

        print(f"End of training epoch {epoch+1}, Dice score: {final_dice_train:.2f}, Loss: {train_loss:.2e}, Avg_loss: {train_loss/cnt_train}")
        
        writer.add_scalar('Training/Final_Dice', final_dice_train, epoch)
        
        writer.add_scalar('Training/Final_Loss', train_loss, epoch)
        writer.add_scalar('Training/Final_Avg_Loss', train_loss/cnt_train, epoch)
        
        # Resetting internal state such that metric is ready for new data
        metric_dice.reset()

            
        #VALIDATION#
        print(" ------------------------------------------------------- Validation ------------------------------------------------------- ")
        model.eval()
        
        start_time_val = time.time()
        val_loss = 0.0
        cnt_val = 0
        
        with torch.no_grad():
            for samples, labels in val_dataloader:
                samples, labels = samples.to(device).float(), labels.to(device).float()
                
                assert samples.type() == labels.type(), f"Samples ({samples.shape}) and labels ({labels.shape}) must have the same type (current: {samples.type()} vs {labels.type()})"
                assert samples.shape[0] == labels.shape[0] and samples.shape[1] == labels.shape[1], f"Samples ({samples.shape}) and labels ({labels.shape}) must have the same number of samples and channels"
            
                out = model(samples)
                    
                out_np = out.cpu().data.numpy()
                labels_np = labels.cpu().data.numpy()
                    
                # compute loss
                loss = criterion(out, labels)
                
                # update validation loss for each batch
                val_loss += loss.item()
                cnt_val += 1
                
                val_dice = metric_dice(out, labels.long()) # per batch
                    
                    
                    
            # End of validation loop 
            final_dice_val = metric_dice.compute()
            new_possible_best = final_dice_val
            
            if evaluation_method.lower() == "dice" and new_possible_best > best_val_dice:
                num_epochs_no_improvement = 0 # reset counter for early stopping
                best_val_dice = new_possible_best # update best validation dice score
                
                # save new best model
                print(f"--- Saving best model at epoch {epoch+1} with validation dice {best_val_dice:.4f}...")
                writer.add_scalar('Validation/Best_dice_score', best_val_dice, epoch)
                torch.save(model.state_dict(), os.path.join(best_model_path,f"best_model.pth"))
                
            elif evaluation_method.lower() == "loss" and (min_val_loss > val_loss): # if validation loss has decreased
                num_epochs_no_improvement = 0 # reset counter for early stopping
                min_val_loss = val_loss # update min validation loss
                # save new best model
                print(f"Saving best model at epoch {epoch+1} with validation loss {min_val_loss:.4f}...")
                writer.add_scalar('Validation/Best_Loss', min_val_loss, epoch)
                torch.save(model.state_dict(), os.path.join(best_model_path,f"best_model.pth"))
                last_best_epoch = epoch+1
            else:
                # increment counter for early stopping
                num_epochs_no_improvement += 1
                if num_epochs_no_improvement >= patience: # if no improvement for patience epochs
                    # stop training - early stopping
                    print(f"Early stopping after {epoch+1} epochs, {num_epochs_no_improvement} without improvement")
                    break
                
            print(f'Epoch [{epoch+1}/{epochs}], Validation Dice score: {(final_dice_val):.2f}, Val-Time: {(time.time() - start_time_val):.2f}s')
            writer.add_scalar('Validation/Dice', final_dice_val, epoch)
            metric_dice.reset()
                
            print(f"End of validation for epoch {epoch+1}, Loss: {val_loss:.2e}, Avg_loss: {val_loss/cnt_val}")
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            writer.add_scalar('Validation/Avg_Loss', val_loss/cnt_val, epoch)
        
        # End of for loop over validation batches 
        if max_steps is not None and cumul_step > max_steps:
            print(f"Max steps reached ({max_steps}). Stopping training...")
            break
            
        model.train()
            
        
    # End of training loop
    if not test_mode:
        writer.close()
        print(f"Training finished after {epoch+1} epochs!")
    else:
        print(f"Training not done!")
        
    print(f"Output dir: {model_filepath}")

    if not test_mode:
        print("Loading best model from training")
    
    datetime_path = datetime_path
    assert datetime_path is not None, "Datetime path must be provided"
    
    best_model = os.path.join(model_filepath,datetime_path,'best_model.pth')
    print(f"Loading best_model.pth")
    assert os.path.exists(best_model), f"Model {best_model} does not exist"

    prediction_filepath = os.path.join(model_filepath,'predictions',dataset_name,datetime_path)
    print(f"Predictions will be saved in {prediction_filepath}")
    mk_dir(prediction_filepath)
    print(f"Best model used for inference:{dataset_name} == {datetime_path}")
    
    #Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Device: {device}")

    model = WNetSeg(n_classes=n_classes, patch_size=patch_size, in_channels=in_channels, activation=nn.ReLU(), final_activation=nn.Sigmoid(), dropout=dropout, bn=bn)
    model.to(device)
    model.load_state_dict(torch.load(best_model))
    model.eval()
    
    labels_filepath = test_set_dir

    print(f"Loading test set from {test_set_dir}...")
    print(f"Loading labels from {labels_filepath} ...")
    
    import sys
    #sys.path.append('/home/falcetta/0_PhD/sparse_var/deep-sparse/patch_dataloader')
    sys.path.append(os.path.join(os.getcwd(),"patch_dataloader"))
    from captcha.active_learning.helper_OOP.dataset import TestSet
    
    test_set = TestSet(test_set_dir)
    input_list = test_set.get_volume_paths()
    mask_list  = test_set.get_brain_paths()

    assert len(input_list) == len(mask_list), f"Number of input and mask files do not match: {len(input_list), len(mask_list)}"
    assert len(input_list)>0, f"No input files found in {test_set_dir}"

    print(f"Test set: {len(input_list)} volumes")

    label_list = test_set.load_true_vessels_paths(labels_filepath)
    print(f"Found {len(label_list)} labels")
        
    assert len(label_list) == len(input_list) or len(label_list) == 0, f"Number of input and label files do not match: {len(input_list), len(label_list)}, {input_list}, {label_list}"
    assert len(label_list)>0, f"No label files found in {labels_filepath}"
    
    
    print(f"---- Volume ---- ---- Mask ----  ---- Label ----")
    for img, mask, label in zip(input_list, mask_list, label_list):
        img = os.path.basename(img)
        mask = os.path.basename(mask)
        label = os.path.basename(label)
        print(f"{img:>15}|{mask:>15}|{label:>15}")
        
    compute_dice = True # if there are no label files, do not compute the dice (default: False)    
    metric_dice = get_metric_dice()
    dice_dict = {}
    cldice_dict = {}
    
    
    from tqdm import tqdm
    import time

    from captcha.active_learning.helper_OOP.patch import MedicalImagePatch
    from captcha.active_learning.helper_OOP.slice import MedicalImageSlice
    from captcha.active_learning.helper_OOP.volume import NiftiVolume
    from captcha.utils.helper import format_duration

    
    overlaps = [0.5]
    PNET_selections = [False] # IF YOU HAVE TIME, TRY TO RUN THE PNET SELECTION
    
    for overlap in overlaps:
        for PNET_selection in PNET_selections:
            threshold = 0.5
            
            print("\n\n############################################################################################################")
            print(f"Running final TESTING for {dataset_name}")   
            print(f"Overlap: {overlap}, PNET: {PNET_selection}, Threshold: {threshold}")
            print("############################################################################################################\n\n")
            time.sleep(2)
            
            PNET_title = 'PNET' if PNET_selection else 'NO_PNET'
            overlap_title = f'{int(overlap*100)}_OVERLAP' if overlap >0 else 'NO_OVERLAP'
            pred_title =  f'{PNET_title}_{overlap_title}'
            
            print(f"Prediction title: {pred_title}")

            prediction_filepath_fin = os.path.join(prediction_filepath,pred_title)
            mk_dir(prediction_filepath_fin)
            
            results_filepath_check = os.path.join(prediction_filepath_fin, f'results_{pred_title}.txt')
            print(f"Results filepath check: {results_filepath_check}")
            
            if os.path.exists(results_filepath_check):
                print(f"Prediction already done for {pred_title}. Skipping prediction...")
                print(f"Results already saved to: {results_filepath_check}")
                continue
            
            # load image, mask and label stacks as matrices
            for i, img_path in enumerate(input_list):
                
                # load image and mask matrices
                current_volume = NiftiVolume(img_path, mask_list[i], label_list[i] if compute_dice else None)
                current_volume.load_data()
                patient_filename = os.path.basename(img_path)
                print(f"Patient: {patient_filename} (#{i+1}/{len(input_list)})")

                
                print("-"*50)
                # skull stripping
                current_volume.skull_strip(current_volume.get_min())
                current_volume.standardize()
                
                # Shape example = 512, 512, 99
                
                print("-"*50)
                # z coordinates where there is brain (z_slice contains the indexes of the slices with brain)
                z, z_slices = current_volume.get_brain_z_slices_idx()
                
                
                # start cutting out and predicting the patches
                starttime_total = time.time()
                print(f"Predicting patches...")
                print("-"*50)
                
                if not current_volume.is_already_predicted(prediction_filepath_fin): 
                    # proceed slice by slice
                    for slice_id in tqdm(z_slices, leave=False, desc="Slices"):
                        
                        #print(f'Slice: #{l}')
                        starttime_slice = time.time()
                        
                        slice_l, label_l, brain_l = current_volume.get_slice(axis=2, index=slice_id, get_brain=True)
                        
                        current_slice = MedicalImageSlice(slice_l, slice_id, vessel_data=label_l, brain_data=brain_l)
                        patches, vessel_patches,  _  ,patch_positions = current_slice.get_consecutive_patches(patch_size=patch_size, overlap_x=overlap, overlap_y=overlap, get_mask_patches=True)
                        
                        modified_patches = np.zeros_like(patches)
                        
                        # ---------------------EXPERIMENTING !!! ----------------------------------------------------------- #
                        #print(f"Predicting TOT {len(patches)} patches... (patch_size: {patches[0].shape})")
                        
                        n=1000
                        patch_chunks = chunked_array(patches, n)
                        vessel_patch_chunks = chunked_array(vessel_patches, n)
                        patch_position_chunks = chunked_iterable(patch_positions, n)

                        for ix, (patch_chunk, vessel_patch_chunk, patch_position_chunk) in enumerate(zip(patch_chunks, vessel_patch_chunks, patch_position_chunks)):
                            patch_chunk = np.array(patch_chunk)
                            vessel_patch_chunk = np.array(vessel_patch_chunk)
                            current_patch = MedicalImagePatch(patch_chunk, vessel_patch_chunk, patch_position_chunk, patch_size, device)
                            current_patch.normalize(metadata) # Normalization with metadata + standardization min-max
                            current_patch.predict_patch(model=model, PNET_selection=PNET_selection) # PNet selection allowa to skip the patch if they are not vessel patches
                            current_patch.reshape_and_threhsold_prediction(threshold=threshold)
                            predictions = current_patch.get_prediction()
                            modified_patches[ix*n:(ix+1)*n] = predictions
                        # -------------------------------------------------------------------------------- #
                        predicted_slice = current_slice.reconstruct_slice(modified_patches, patch_positions, method='avg', threshold=0.5)
                        current_volume.reconstruct_volume(predicted_slice, slice_id)
                        # ------------------ End of patch loop over a slice ------------------ #
                else:
                    slice_id = 1
                    print(f"Prediction already saved for {patient_filename}. Skipping prediction...")
                # ------------------ End of slice loop over a patient ------------------ #

                # how long does the prediction take for a patient
                duration_total = time.time() - starttime_total
                patient_time = format_duration(duration_total)
                print(f'Prediction for a patient ({i+1}/{len(input_list)}) took: {patient_time} (Avg: {duration_total/slice_id:.2f}s per slice)')
                print(f"Metrics: [Dice, CLDice]")
                
                print(f'Prediction saved to: {prediction_filepath_fin}')
                _ , current_dice , current_cldice = current_volume.save_predicted_volume(prediction_filepath_fin, compute_dice=compute_dice)
                dice_dict[current_volume.get_id()] = current_dice
                cldice_dict[current_volume.get_id()] = current_cldice
                
                print("-"*50)
                
            # ----------------------------------------------- End of patient loop ----------------------------------------------- #

            print(f"WNets prediction finished.")
                
            # ------------------ End of main function ------------------ #

    
            print(f"Results for {dataset_name} dataset")
            print(f"|  ID  |  DICE  | CLDICE |")
            for patient_key, dice_score in dice_dict.items():
                print(f'| {patient_key:>4} |  {dice_score:5.2f} | {cldice_dict[patient_key]:6.2f} |')

            avg_dice = sum(dice_dict.values())/len(dice_dict)
            avg_cldice = sum(cldice_dict.values())/len(cldice_dict)
            std_dice = np.std(list(dice_dict.values()))
            std_cldice = np.std(list(cldice_dict.values()))

            print(f'|  AVG | {avg_dice:6.2f} | {avg_cldice:6.2f} |')
            print(f'|  STD | {std_dice:6.2f} | {std_cldice:6.2f} |')


            results_filepath = os.path.join(prediction_filepath_fin, f'results_{pred_title}.txt')
            with open(results_filepath, "w") as file:
                file.write(f"Results for {dataset_name} dataset\n")
                file.write(f"|  ID  |  DICE  | CLDICE |\n")
                for patient_key, dice_score in dice_dict.items():
                    file.write(f'| {patient_key:>4} |  {dice_score:5.2f} | {cldice_dict[patient_key]:6.2f} |\n')

                avg_dice = sum(dice_dict.values()) / len(dice_dict)
                avg_cldice = sum(cldice_dict.values()) / len(cldice_dict)
                std_dice = np.std(list(dice_dict.values()))
                std_cldice = np.std(list(cldice_dict.values()))

                file.write(f'|  AVG | {avg_dice:6.2f} | {avg_cldice:6.2f} |\n')
                file.write(f'|  STD | {std_dice:6.2f} | {std_cldice:6.2f} |\n')

            print(f"Results saved to {results_filepath}")
    
    
    
    
if __name__ == "__main__":
    from torch.utils.data import Dataset

    class WNetPatchDataset(Dataset):
        
        def __init__(self, images_array, labels_array, transform=None, metadata=None):
            self.images_array = images_array
            self.labels_array = labels_array
            self.transform = transform
            
            if metadata is not None:
                self.mean, self.std = metadata
            else:
                self.compute_metadata()
            
            
        def __getitem__(self,index):
            # Select random index from the dataset
            idx_0 = int(index)
            img0, label0 = self.images_array[idx_0], self.labels_array[idx_0]
            
            # Apply transformations if provided
            if self.transform is not None:
                img0 = self.transform(img0)
                label0 = self.transform(label0)
            
            # Normalize images
            img0 = (img0 - self.mean) / self.std    
            # Standardize to 0-1
            img0 = (img0 - img0.min()) / (img0.max() - img0.min())
            
            assert min(img0.flatten()) >= 0, f"Min value: {min(img0.flatten())}"
            assert max(img0.flatten()) <= 1, f"Max value: {max(img0.flatten())}"
            
            return img0, label0
            
        def compute_metadata(self):
            flattened_data = self.images_array.reshape(self.images_array.shape[0], -1)
            means = np.mean(flattened_data, axis=0)
            stds = np.std(flattened_data, axis=0)
            
            # compute the single mean and std
            mean = np.mean(means)
            std = np.mean(stds)
            
            #convert to pytorch
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std)
            
            print(f"Mean: {self.mean} - Std: {self.std}")

        def get_metadata(self):
            return self.mean, self.std
        
        def __len__(self):
            return len(self.images_array)
        


    class UpConcatBlock(nn.Module):
        
        """
        Bulding block with up-sampling and concatenation for one level in the first 2D-Unet.
        """
        
        def __init__(self, pool_size, concat_axis):
            super(UpConcatBlock, self).__init__()

            self.up = nn.Upsample(scale_factor=pool_size, mode='bilinear', align_corners=True)
            self.concat_axis = concat_axis

        def forward(self, x1, x2):
            x1 = self.up(x1)
            #print(f"Concat: {x1.shape, x2.shape} along axis {self.concat_axis}")
            out = torch.cat([x1, x2], dim=self.concat_axis)
            return out

    class UpConcatBlock2(nn.Module):
        
        """
        Bulding block with up-sampling and concatenation for one level in the second 2D-Unet.
        """
        
        def __init__(self, pool_size, concat_axis):
            super(UpConcatBlock2, self).__init__()

            self.up = nn.Upsample(scale_factor=pool_size, mode='bilinear', align_corners=True)
            self.concat_axis = concat_axis

        def forward(self, x1, x2, x3):
            x1 = self.up(x1)
            #print(f"Concat {x1.shape, x2.shape, x3.shape} along axis {self.concat_axis}")
            out = torch.cat([x1, x2, x3], dim=self.concat_axis)
            return out


    class DoublConvBlock(nn.Module):
        
        """
        Bulding block with convolutional layers for one level.
        """
        
        def __init__(self, in_channels, num_kernels, kernel_size, strides, padding, activation, dropout=False, bn=False):
            
            super(DoublConvBlock, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_kernels,
                                kernel_size=kernel_size,stride=strides, padding=padding)
            
            self.activation = activation
            
            self.bn1 = nn.BatchNorm2d(num_kernels) if bn else None
            
            self.dropout = nn.Dropout(p=dropout) if dropout else None
        
            self.conv2 = nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels,
                                kernel_size=kernel_size,stride=strides, padding=padding)
            
            self.bn2 = nn.BatchNorm2d(num_kernels) if bn else None

        
            
        def forward(self, x):
            
            #print("ConvBlock1: ", x.shape)
            x = self.conv1(x)
            
            #print("Activation: ", self.activation)
            x = self.activation(x)
            
            if self.bn1:
                #print("BatchNorm1: ", x.shape)
                x = self.bn1(x)
            
            if self.dropout:
                #print("Dropout: ", x.shape)
                x = self.dropout(x)
            
            #print("ConvBlock2: ", x.shape)
            x = self.conv2(x)
            #print("Activation: ", self.activation)
            
            x = self.activation(x)
            
            if self.bn2:
                #print("BatchNorm2: ", x.shape)
                x = self.bn2(x)
            
            #print("Out: ", x.shape)
            
            return x

    class FinalConvBlock(nn.Module):
        
        """
        Bulding block with convolutional layers for one level.
        """
        
        def __init__(self, in_channels, out_channels, kernel_size, strides, padding, activation, bn=False):
            
            super(FinalConvBlock, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size,stride=strides, padding=padding)
            
            self.activation = activation
            
        
            
        def forward(self, x):
            
            #print("ConvBlock1: ", x.shape)
            x = self.conv1(x)
            
            #print("Activation: ", self.activation)
            x = self.activation(x)
            
            #print("Out: ", x.shape)
            
            return x

    class WNetSeg(nn.Module):    
        
        """
        Defines the architecture of the wnetseg. 
        """
            
        def __init__(self, patch_size, in_channels, activation, final_activation, kernel_size=(3, 3), pool_size=(2, 2), strides=(1, 1), num_kernels=None, concat_axis=1, padding='same', final_padding = 'same', bn=False,  dropout=False, n_classes=1):
            
            super(WNetSeg, self).__init__()

            if num_kernels is None:
                num_kernels = [64, 128, 256, 512, 1024] # Output channels of each level of the U-net

            
            # The first U-net
            # DOWN-SAMPLING PART (left side of the first U-net)
            # layers on each level: convolution2d -> dropout -> convolution2d -> max-pooling
            # level 0
            self.conv_0_down_1 = DoublConvBlock(in_channels, num_kernels[0], kernel_size, strides, padding, activation, dropout, bn)
            self.pool_0_1 = nn.MaxPool2d(kernel_size=pool_size)
            
            # level 1
            self.conv_1_down_1 = DoublConvBlock(num_kernels[0], num_kernels[1], kernel_size, strides, padding, activation, dropout, bn)
            self.pool_1_1 = nn.MaxPool2d(kernel_size=pool_size)                               
        
            # level 2
            self.conv_2_down_1 = DoublConvBlock(num_kernels[1], num_kernels[2], kernel_size, strides, padding, activation, dropout, bn)
            self.pool_2_1 = nn.MaxPool2d(kernel_size=pool_size)                               
        
            # level 3
            self.conv_3_1 = DoublConvBlock(num_kernels[2], num_kernels[3], kernel_size, strides, padding, activation, dropout, bn)
            
            # UP-SAMPLING PART (right side of the first U-net)
            # layers on each level: upsampling2d -> concatenation with feature maps of corresponding level from down-sampling
        
            # level 2
            self.concat_2_1 = UpConcatBlock(pool_size, concat_axis)
            self.conv_2_up_1 = DoublConvBlock(num_kernels[3]+num_kernels[2], num_kernels[2], kernel_size, strides, padding, activation, dropout,bn)
            
            # level 1
            self.concat_1_1 = UpConcatBlock(pool_size, concat_axis)
            self.conv_1_up_1 = DoublConvBlock(num_kernels[2]+num_kernels[1], num_kernels[1], kernel_size, strides, padding, activation, dropout,bn)

            # level 0
            self.concat_0_1 = UpConcatBlock(pool_size, concat_axis)
            self.conv_0_up_1 = DoublConvBlock(num_kernels[1]+num_kernels[0], num_kernels[0], kernel_size, strides, padding, activation, dropout,bn)
            
            self.final_conv_1 = FinalConvBlock(num_kernels[0], n_classes, 1, strides, padding=final_padding, activation=final_activation)
            
            # The second U-net
            # DOWN-SAMPLING PART (left side of the second U-net)
            
            # level 0
            self.conv_0_down_2 = DoublConvBlock(n_classes, num_kernels[0], kernel_size, strides, padding, activation, dropout, bn)
            self.pool_0_2 = nn.MaxPool2d(kernel_size=pool_size)

            # level 1
            self.conv_1_down_2 = DoublConvBlock(num_kernels[0], num_kernels[1], kernel_size, strides, padding, activation, dropout, bn)
            self.pool_1_2 = nn.MaxPool2d(kernel_size=pool_size)                               
        
            # level 2
            self.conv_2_down_2 = DoublConvBlock(num_kernels[1], num_kernels[2], kernel_size, strides, padding, activation, dropout, bn)
            self.pool_2_2 = nn.MaxPool2d(kernel_size=pool_size)                               
        
            # level 3
            self.conv_3_2 = DoublConvBlock(num_kernels[2], num_kernels[3], kernel_size, strides, padding, activation, dropout, bn)
            
            # UP-SAMPLING PART (right side of the second U-net)
            
            # level 2
            self.concat_2_2 = UpConcatBlock2(pool_size, concat_axis)
            self.conv_2_up_2 = DoublConvBlock(num_kernels[3]+2*num_kernels[2], num_kernels[2], kernel_size, strides, padding, activation, dropout,bn)
            # level 1
            self.concat_1_2 = UpConcatBlock2(pool_size, concat_axis)
            self.conv_1_up_2 = DoublConvBlock(num_kernels[2]+2*num_kernels[1], num_kernels[1], kernel_size, strides, padding, activation, dropout,bn)

            # level 0
            self.concat_0_2 = UpConcatBlock2(pool_size, concat_axis)
            self.conv_0_up_2 = DoublConvBlock(num_kernels[1]+2*num_kernels[0], num_kernels[0], kernel_size, strides, padding, activation, dropout,bn)
            
            self.final_conv_2 = FinalConvBlock(num_kernels[0], n_classes, 1, strides, padding=final_padding, activation=final_activation)
        
        def forward(self, x):
            
            # The first U-net
            #print("\n\n --- First U-net ---")
            # DOWN-SAMPLING PART (left side of the first U-net)
            
            #save_img(x, 'input.png')            
            
            # level 0
            #print("\n\n --- Level 0\n")
            x1 = self.conv_0_down_1(x)
            x2 = self.pool_0_1(x1)
            
            # level 1
            #print("\n\n --- Level 1\n")
            x3 = self.conv_1_down_1(x2)
            x4 = self.pool_1_1(x3)
            
            # level 2
            #print("\n\n --- Level 2\n")
            x5 = self.conv_2_down_1(x4)
            x6 = self.pool_2_1(x5)
            
            # level 3
            #print("\n\n --- Level 3\n")
            x7 = self.conv_3_1(x6)
            
            # UP-SAMPLING PART (right side of the first U-net)
            
            # level 2
            #print("\n\n --- Level 2\n")
            x8 = self.concat_2_1(x7,x5)
            x9 = self.conv_2_up_1(x8)
            
            # level 1
            #print("\n\n --- Level 1\n")
            x10 = self.concat_1_1(x9,x3)
            x11 = self.conv_1_up_1(x10)
            
            # level 0
            #print("\n\n --- Level 0\n")
            x12 = self.concat_0_1(x11,x1)
            x13 = self.conv_0_up_1(x12)
            
            #print("\n\n --- Final Conv\n")
            output_1 = self.final_conv_1(x13)
            
            # The Second U-net
            #print("\n\n --- Second U-net ---")
            # DOWN-SAMPLING PART (left side of the second U-net)
            #save_img(output_1, 'output_1.png')
            
            # level 0
            #print("\n\n --- Level 0\n")
            y1 = self.conv_0_down_2(output_1)
            y2 = self.pool_0_2(y1)
            
            # level 1
            #print("\n\n --- Level 1\n")
            y3 = self.conv_1_down_2(y2)
            y4 = self.pool_1_2(y3)
            
            # level 2
            #print("\n\n --- Level 2\n")
            y5 = self.conv_2_down_2(y4)
            y6 = self.pool_2_2(y5)
            
            # level 3
            #print("\n\n --- Level 3\n")
            y7 = self.conv_3_2(y6)
            

            # UP-SAMPLING PART (right side of the second U-net)
            
            # level 2
            #print("\n\n --- Level 2\n")
            y8 = self.concat_2_2(y7,x5,y5)
            y9 = self.conv_2_up_2(y8)
            
            # level 1
            #print("\n\n --- Level 1\n")
            y10 = self.concat_1_2(y9,x3,y3)
            y11 = self.conv_1_up_2(y10)
            
            # level 0
            #print("\n\n --- Level 0\n")
            y12 = self.concat_0_2(y11,x1,y1)
            y13 = self.conv_0_up_2(y12)
            
            #print("\n\n --- Final Conv\n")
            output_2 = self.final_conv_2(y13)
            #save_img(output_2, 'output_2.png')
            
            return output_2
        

    def soft_erode(img):
        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(img):
        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


    def soft_open(img):
        return soft_dilate(soft_erode(img))

    def soft_skel(img, iter_):
        img1  =  soft_open(img)
        skel  =  F.relu(img-img1)
        for j in range(iter_):
            img  =  soft_erode(img)
            img1  =  soft_open(img)
            delta  =  F.relu(img-img1)
            skel  =  skel +  F.relu(delta-skel*delta)
        return skel

    class soft_cldice(nn.Module):
        def __init__(self, iter_=3, smooth = 1.):
            super(soft_cldice, self).__init__()
            self.iter = iter_
            self.smooth = smooth

        def forward(self, y_true, y_pred):
            skel_pred = soft_skel(y_pred, self.iter)
            skel_true = soft_skel(y_true, self.iter)
            
            tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)    
            tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
            
            cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
            
            return cl_dice


    def soft_dice(y_true, y_pred):
        """[function to compute dice loss]

        Args:
            y_true ([float32]): [ground truth image]
            y_pred ([float32]): [predicted image]

        Returns:
            [float32]: [loss value]
        """
        smooth = 1
        intersection = torch.sum((y_true * y_pred), dim=(0,2,3))
        coeff = (2. *  intersection + smooth) / (torch.sum(y_true, dim=(0,2,3) ) + torch.sum(y_pred, dim=(0,2,3)) + smooth)
        return (1. - coeff)


    class soft_dice_cldice(nn.Module):
        def __init__(self, iter_=3, alpha=0.5, smooth = 1.e-5):
            super(soft_dice_cldice, self).__init__()
            self.iter = iter_
            self.smooth = smooth
            self.alpha = alpha

        def forward(self, y_pred, y_true, ignore_background=False):
            dice = soft_dice(y_true, y_pred)
            
            skel_pred = soft_skel(y_pred, self.iter)
            skel_true = soft_skel(y_true, self.iter)
            
            tprec = (torch.sum(torch.multiply(skel_pred, y_true), dim=(0,2,3))+self.smooth)/(torch.sum(skel_pred, dim=(0,2,3))+self.smooth)    
            tsens = (torch.sum(torch.multiply(skel_true, y_pred), dim=(0,2,3))+self.smooth)/(torch.sum(skel_true, dim=(0,2,3))+self.smooth)    
            
            cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
            
            return (1.0-self.alpha)*dice+self.alpha*cl_dice

    def GDiceLoss(y_true, y_pred, ignore_background=True, weighed=False):
        """
        Generalized Dice Loss
        
        Input:
            y_true: ground truth mask [batch_size, num_classes, height, width] (num_classe could also be 1)
            y_pred: predicted mask [batch_size, num_classes, height, width]
        Output:
            dice_score: generalized dice loss
        """
        assert y_true.shape == y_pred.shape, f"y_true and y_pred must have the same shape {y_true.shape} vs {y_pred.shape}"
        # print("Shape of loss vector", y_true.shape) #16,1,96,96
        
        tp = torch.sum(y_true * y_pred, dim=(0,2,3))
        fp = torch.sum(y_true*(1-y_pred),dim=(0,2,3))
        fn = torch.sum((1-y_true)*y_pred,dim=(0,2,3))
        
        nominator = 2*tp + 1e-05
        denominator = 2*tp + fp + fn + 1e-9
        
        if ignore_background and y_true.shape[1] > 1:
            #print("Ignoring background class")
            dice_score = 1 -(nominator / (denominator+1e-9))[1:]
        else:
            #print("Considering all classes")
            dice_score = 1 -(nominator / (denominator+1e-9))
        
        # ------------------------ # Weighted Dice Loss # ------------------------ #
        # TODO: add weights for each class (It depends on the dataset and number of classes)
        if weighed:
            weighed_dice_score = torch.tensor([0.1, 0.9]).cuda()
            dice_score = torch.mean(weighed_dice_score * dice_score)
        # ------------------------# -------------------- # ------------------------ #
        else:
            dice_score = torch.mean(dice_score) # Average over all classes (or all classes except background)
            
        return dice_score
    main()
    
    