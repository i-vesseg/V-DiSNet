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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torchmetrics
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from wnet_dataset_utils import WNetPatchDataset
from wnet_utils import WNetSeg
from seg_utils import GDiceLoss, soft_dice_cldice

from utils import mk_dir
from sampler import get_samples, get_SOTA_samples

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
    
    #n_size = 5/100 # 0.1/100, 1/100, 5/100, 10/100, 30/100, 50/100, 75/100
    take = 'all'
    
    # for dataset_name in ['C', 'CF', 'CFX']: # My baselines
    #     for n_size in [1/100, 0.1/100, 5/100, 10/100, 30/100, 50/100, 75/100]:
    #         # choose 8 random seeds between 0 and 1000
    #         for current_seed in [1, 42, 97, 123, 256, 333, 555, 666]:
    #             run_final_experiment(test_mode, dataset_name, n_size, take, current_seed)
                
    #         if dataset_name == 'all':
    #             break

    for dataset_name in ['all', 'R']: # RANDOM and COMPLETE DATASET
        for n_size in [1/100, 0.1/100, 5/100, 10/100, 30/100, 50/100, 75/100]:
            # choose 8 random seeds between 0 and 1000
            for current_seed in [1, 42, 97, 123, 256, 333, 555, 666]:
                run_final_experiment(test_mode, dataset_name, n_size, take, current_seed)
                
            if dataset_name == 'all':
                break
    
    for dataset_name in ['SOTA_AET','SOTA_CA', 'SOTA_RA']: # Sota baselines
        for n_size in [1/100, 0.1/100, 5/100, 10/100, 30/100, 50/100, 75/100]:
            # choose 8 random seeds between 0 and 1000
            for current_seed in [1, 42, 97, 123, 256, 333, 555, 666]:
                run_final_experiment(test_mode, dataset_name, n_size, take, current_seed)
                
            if dataset_name == 'all':
                break
    


def run_final_experiment(test_mode, dataset_name, n_size, take, current_seed, original_dataset='CAS'):
    
    # -------------------------- Load Metadata --------------------------
    save_embeddings_path = '/data/falcetta/brain_data/embeddings_VDISNET'

    print("\n### ------------------------------------------------------------------- ###")
    print(f"Running final experiment for Dataset:{dataset_name} - Size: {n_size} - {take} - Seed:{current_seed}")
    print("### ------------------------------------------------------------------- ###\n")
    
    print(f"Loading training patches from {save_embeddings_path}")
    print("Loading metadata from the complete dataset")
    metadata = np.load(os.path.join(save_embeddings_path, f'metadata_{original_dataset}_all.npy'))

                
    print(f"Metadata {metadata}")
    # ----------------------------------------------------- #
    if dataset_name == 'R':
        print("Loading Random R dataset")
    
    if dataset_name == 'C':
        print("Loading Class-sampled C dataset")
    
    if dataset_name == 'CF':
        print("Loading Class-sampled CF dataset")
    
    if dataset_name == 'CFX':
        print("Loading Class-sampled CFX dataset")
    else:
        print(f"Loading {dataset_name} dataset")
    
    if 'SOTA' in dataset_name:
        X_test_sampled, X_test_mask_sampled = get_SOTA_samples(save_embeddings_path, dataset_name, n_size, take, current_seed, original_dataset)
    else:
        X_test_sampled, X_test_mask_sampled = get_samples(save_embeddings_path, dataset_name, n_size, take, current_seed, original_dataset)
    
    # ----------------------------------------------------- #
        
    print(f"Loading dataset {dataset_name} - {n_size} - {take}")
    
    X, X_mask = X_test_sampled, X_test_mask_sampled
    print(f"Dataset name: {dataset_name}")

    print(f"Dataset for train: {X.shape}, {X_mask.shape}")
    
    #DATA RETRIVAL
    model_filepath = f'/data/falcetta/brain_data/FINAL_EXP_VDISNET_JOURNAL_SEEDs/{n_size}/{dataset_name}/seed_{current_seed}'
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
        max_steps = 15000

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
                
                
            #     start_time = time.time() 
            
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
    
    test_set_dir = "/data/falcetta/brain_data/CASJ/preprocessed/numpy/test"

    labels_filepath = test_set_dir

    print(f"Loading test set from {test_set_dir}...")
    print(f"Loading labels from {labels_filepath} ...")
    
    import sys
    sys.path.append('/home/falcetta/0_PhD/sparse_var/deep-sparse/patch_dataloader')
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
            print(f"Running final TESTING for {dataset_name} - {n_size} - {take}")   
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
                aaa = current_volume.load_data()
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
                            #print(f"\n\n\nPredicting {len(patch_chunk)} patches...")
                            #print(len(patch_chunk), len(vessel_patch_chunk), len(patch_position_chunk))
                            #print(patch_chunk[0].shape)
                            #print(vessel_patch_chunk[0].shape)
                            #print(patch_position_chunk[0])
                            patch_chunk = np.array(patch_chunk)
                            vessel_patch_chunk = np.array(vessel_patch_chunk)
                            current_patch = MedicalImagePatch(patch_chunk, vessel_patch_chunk, patch_position_chunk, patch_size, device)
                            current_patch.normalize(metadata) # Normalization with metadata + standardization min-max
                            current_patch.predict_patch(model=model, PNET_selection=PNET_selection) # PNet selection allowa to skip the patch if they are not vessel patches
                            current_patch.reshape_and_threhsold_prediction(threshold=threshold)
                            predictions = current_patch.get_prediction()
                            modified_patches[ix*n:(ix+1)*n] = predictions
                        # -------------------------------------------------------------------------------- #
                        
                        # -----------------OLD FOR--------------------------------------------------------------- #
                        # for patch, vessel_patch, patch_position in zip(patches, vessel_patches, patch_positions):
                        #     #print("\n\n\nPredicting a patch...")
                        #     #print(patch.shape) # (32, 32)
                        #     #print(vessel_patch.shape) # (32, 32)
                        #     #print(patch_position) # (64,96,240,272) ==> Position x,y
                        #     current_patch = MedicalImagePatch(patch, vessel_patch, patch_position, patch_size, device)
                           
                        #     current_patch.normalize(metadata) # Normalization with metadata + standardization min-max
                        #     current_patch.predict_patch(model=model, PNET_selection=PNET_selection) # PNet selection allowa to skip the patch if they are not vessel patches

                        #     current_patch.reshape_and_threhsold_prediction(threshold=threshold)
                        #     modified_patches.append(current_patch.get_prediction())
                        # # -------------------------------------------------------------------------------- #
                        
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
    main()
    
    