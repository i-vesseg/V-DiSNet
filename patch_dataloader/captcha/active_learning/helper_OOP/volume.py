import os
import numpy as np
import nibabel as nib
from sklearn.metrics import precision_score, recall_score

class NiftiVolume():
    def __init__(self, filepath, brain_filepath=None ,vessel_filepath=None, vessel_filepath_2=None):
        self.filepath = filepath
        
        self.vessel_filepath = vessel_filepath
        self.vessel_filepath_2 = vessel_filepath_2
        self.brain_filepath = brain_filepath
        self.reconstructed_volume = None
        self.volume = None
        self.id = None
        self.vessel_volume = None
        self.vessel_volume_2 = None
        self.brain_volume = None
        

    def load_data(self):
        #print(f' - Loading volume from {self.filepath}')
        self.load_volume()
        
        if self.vessel_filepath is not None:
            #print(f' - Loading vessel volume from {self.vessel_filepath}')
            self.load_vessel_volume()
            assert self.volume.shape == self.vessel_volume.shape, f'{self.volume.shape} != {self.vessel_volume.shape}'
        
        if self.vessel_filepath_2 is not None:
            #print(f' - Loading vessel volume from {self.vessel_filepath_2}')
            self.load_vessel_volume_2()
            assert self.volume.shape == self.vessel_volume_2.shape, f'{self.volume.shape} != {self.vessel_volume_2.shape}'
        
        if self.brain_filepath is not None:
            #print(f' - Loading brain volume from {self.brain_filepath}')
            self.load_brain_volume()
            assert self.volume.shape == self.brain_volume.shape, f'{self.volume.shape} != {self.brain_volume.shape}'
        
        assert self.volume is not None, "Volume is not loaded"
        
    def get_id(self):
        self.id = str(self.filepath.split(os.sep)[-1].split('_')[0]) if self.id is None else str(self.id)
        self.check_id()
        
        return self.id

    def check_id(self):
        if self.id is None:
            raise ValueError("Id is not set")
        
        assert self.id.isdigit(), f'Id is not a digit: {self.id} for item: {self.filepath}'
        assert len(self.id) <= 3, f'Id is not max 3 digits: {self.id} for item: {self.filepath}' 
        
        if self.vessel_filepath is not None:
            assert self.id == str(self.vessel_filepath.split(os.sep)[-1].split('_')[0]), f'Id mismatch: {self.id} != {self.vessel_filepath.split(os.sep)[-1].split("_")[0]}'
        if self.brain_filepath is not None:
            assert self.id == str(self.brain_filepath.split(os.sep)[-1].split('_')[0]), f'Id mismatch: {self.id} != {self.brain_filepath.split(os.sep)[-1].split("_")[0]}'
        #print(f' - Id: {self.id}')
        
    def get_volume(self):
        return self.volume
    
    def get_data(self):
        if self.vessel_volume_2 is not None:
            return self.volume, self.brain_volume, self.vessel_volume, self.vessel_volume_2
        else:
            return self.volume, self.brain_volume, self.vessel_volume 
    
    def get_min(self):
        min_volume = np.min(self.volume)
        #print(f'Minimum value: {min_volume}')
        return min_volume
    
    def get_max(self):
        max_volume = np.max(self.volume)
        #print(f'Maximum value: {max_volume}')
        return max_volume
    
    def load_volume(self):
        nifti = nib.load(self.filepath)
        self.volume = nifti.get_fdata()

    def load_vessel_volume(self, vessel_filepath=None):
        self.vessel_filepath = vessel_filepath if vessel_filepath is not None else self.vessel_filepath
        if self.vessel_filepath is None:
            raise ValueError("No label volume is specified")
        
        vessel_nifti = nib.load(self.vessel_filepath)
        self.vessel_volume = vessel_nifti.get_fdata()
        self.reconstructed_volume = np.zeros_like(self.vessel_volume)
    
    def load_vessel_volume_2(self, vessel_filepath_2=None):
        self.vessel_filepath_2 = vessel_filepath_2 if vessel_filepath_2 is not None else self.vessel_filepath_2
        if self.vessel_filepath_2 is None:
            raise ValueError("No label volume is specified")
        
        vessel_nifti = nib.load(self.vessel_filepath_2)
        self.vessel_volume_2 = vessel_nifti.get_fdata()
        self.reconstructed_volume_2 = np.zeros_like(self.vessel_volume_2)
        
    def get_shape(self):
        if self.brain_volume is not None:
            assert self.volume.shape == self.brain_volume.shape, f'{self.volume.shape} != {self.brain_volume.shape}'
        if self.vessel_volume is not None:
            assert self.volume.shape == self.vessel_volume.shape, f'{self.volume.shape} != {self.vessel_volume.shape}'
        return self.volume.shape
        
    def load_brain_volume(self):
        if self.brain_filepath is None:
            raise ValueError("No brain volume is specified")
        
        brain_nifti = nib.load(self.brain_filepath)
        self.brain_volume = brain_nifti.get_fdata()
    
    def save_all_volumes(self, filepath):
        self.save_volume(os.path.join(filepath, f'{self.id}_img.nii.gz'))
        self.save_vessel_volume(os.path.join(filepath, f'{self.id}_label.nii.gz'))
        self.save_brain_volume(os.path.join(filepath, f'{self.id}_mask.nii.gz'))
        
    def save_volume(self, filepath):
        nifti = nib.Nifti1Image(self.volume, np.eye(4))
        nib.save(nifti, filepath)
    
    def save_vessel_volume(self, filepath):
        if self.vessel_volume is None:
            raise ValueError("No label volume is specified")
        
        nifti = nib.Nifti1Image(self.vessel_volume, np.eye(4))
        nib.save(nifti, filepath)
    
    def save_brain_volume(self, filepath):
        if self.brain_volume is None:
            raise ValueError("No brain volume is specified")
        
        nifti = nib.Nifti1Image(self.brain_volume, np.eye(4))
        nib.save(nifti, filepath)
        
    def skull_strip(self, bg_value=0, mask_vessel=True):
        
        if self.brain_volume is None:
            raise ValueError("No brain volume is specified")
        
        print(f' - Applying mask to the volume with background value: {bg_value}')
        masked_volume = self.volume.copy()
        masked_volume[np.where(self.brain_volume == 0)] = bg_value
        self.volume = masked_volume
        
        if self.vessel_volume is not None and mask_vessel:
            print(f' - Applying mask to the vessels with background value: 0')
            masked_vessel = self.vessel_volume.copy()
            masked_vessel[np.where(self.brain_volume == 0)] = 0
            self.vessel_volume = masked_vessel

    def standardize(self, mean=None, std=None):
        print(f' - Standardizing the volume')
        mean = np.mean(self.volume) if mean is None else mean
        std = np.std(self.volume) if std is None else std
        self.volume = (self.volume - mean) / std
        self.volume = (self.volume - np.min(self.volume)) / (np.max(self.volume) - np.min(self.volume))
        return self.volume
        
    def enhance_vessels_3d(self):
        print(f"Max and min: {self.volume.max()}, {self.volume.min()} (ORIGINAL)")
        #fig, axs = plt.subplots(1,2, figsize=(10, 5))
        # Plot histogram of all values
        #axs[0].hist(volume.flatten(), bins=100)
        # Plot histogram of positive values
        #axs[1].hist(volume[volume > 0].flatten(), bins=100)
        #plt.show()
        
        # threshold 98% of the values
        threshold = 0.78 * self.volume.max()
        print(f"Threshold: {threshold}")
        enhanced_volume = np.where(self.volume > threshold, 5*threshold, self.volume.min())
        self.volume = enhanced_volume
        
    def get_slice(self, axis=2, index=0, get_brain=False):
        if axis == 0:
            slice_volume = self.volume[index, :, :]
            slice_vessel_volume = self.vessel_volume[index, :, :] if self.vessel_volume is not None else None
            slice_brain_volume = self.brain_volume[index, :, :] if self.brain_volume is not None else None
        elif axis == 1:
            slice_volume = self.volume[:, index, :]
            slice_vessel_volume = self.vessel_volume[:, index, :] if self.vessel_volume is not None else None
            slice_brain_volume = self.brain_volume[:, index, :] if self.brain_volume is not None else None
        elif axis == 2:
            slice_volume = self.volume[:, :, index]
            slice_vessel_volume = self.vessel_volume[:, :, index] if self.vessel_volume is not None else None
            slice_brain_volume = self.brain_volume[:, :, index] if self.brain_volume is not None else None
        else:
            raise ValueError("Invalid axis index")

        if slice_vessel_volume is None:
          print(f"WARN: No label volume is specified for slice {index} along axis {axis}")  
        
        if get_brain:
            return slice_volume, slice_vessel_volume, slice_brain_volume
        
        return slice_volume, slice_vessel_volume
    
            
            
    def get_brain_coord(self):
        if self.brain_volume is None:
            raise ValueError("No brain volume is specified")
        
        self.brain_coord = np.where(self.brain_volume)
        return self.brain_coord
    
    def get_brain_z_slices_idx(self):
        if self.brain_volume is None:
            raise ValueError("No brain volume is specified")
        
        self.get_brain_coord()
        z = self.brain_coord[-1]
        
        return z, np.unique(z)
    
    def reconstruct_volume(self, slice, slice_id):
        self.reconstructed_volume[:, :, slice_id] = slice
    
    def get_reconstructed_volume(self):
        return self.reconstructed_volume

    def compute_dice(self):
        if self.vessel_volume is None:
            raise ValueError("No label volume is specified")
        
        def dc(result, reference):
            result = np.atleast_1d(result.astype(bool))
            reference = np.atleast_1d(reference.astype(bool))
            
            intersection = np.count_nonzero(result & reference)
            
            size_i1 = np.count_nonzero(result)
            size_i2 = np.count_nonzero(reference)
            
            try:
                dc = 2. * intersection / float(size_i1 + size_i2)
            except ZeroDivisionError:
                dc = 0.0
            
            return dc
        
        self.dice = dc(self.vessel_volume, self.reconstructed_volume)
        print(f' - -- Dice score: {self.dice}')
        return self.dice
    
    def compute_cl_dice(self):
        if self.vessel_volume is None:
            raise ValueError("No label volume is specified")
        
        self.cl_dice = clDice(self.vessel_volume, self.reconstructed_volume)
        print(f' - CL Dice score: {self.cl_dice}')

        return self.cl_dice
    
    def compute_cls_metrics(self): #TODO: check this
        if self.vessel_volume is None:
            raise ValueError("No label volume is specified")
        
        self.recall = recall_score(self.vessel_volume.flatten(), self.reconstructed_volume.flatten())
        self.precision = precision_score(self.vessel_volume.flatten(), self.reconstructed_volume.flatten())
        
        print(f' - Recall score: {self.recall}')
        print(f' - Precision score: {self.precision}')
    
    def is_already_predicted(self, filepath):
        if self.id is None:
            self.get_id()
        is_already_predicted_bool = os.path.isfile(os.path.join(filepath, f'prediction_{self.id}.nii.gz'))
        print(f' - Checking if prediction is already saved: {is_already_predicted_bool} ({os.path.join(filepath, f"prediction_{self.id}.nii.gz")})')
        return is_already_predicted_bool
        
        
    def save_predicted_volume(self, filepath,compute_dice=False, compute_acc=False, skull_strip=True):
        self.get_id()
        
        if not self.is_already_predicted(filepath):
            if skull_strip:
                print("Skull stripping on final reconstructed volume")
                self.reconstructed_volume = np.where(self.brain_volume > 0.5, self.reconstructed_volume, 0) 
                    
            nifti = nib.Nifti1Image(self.reconstructed_volume, np.eye(4))
            nib.save(nifti, os.path.join(filepath, f'prediction_{self.id}.nii.gz'))
        else:
            self.reconstructed_volume = nib.load(os.path.join(filepath, f'prediction_{self.id}.nii.gz')).get_fdata()
            
        if compute_dice:
            self.compute_dice()
            self.compute_cl_dice()
        if compute_acc:
            self.compute_cls_metrics()
        
        
        return self.reconstructed_volume, self.dice, self.cl_dice
    
    # ---------------------------------- GRID ----------------------------------
    def create_empty_grid(self, as_type=None):
        self.grid = np.zeros_like(self.volume)
        if as_type:
            self.grid = self.grid.astype(as_type)
    
    def reconstruct_grid(self, slice, slice_id):
        self.grid[:, :, slice_id] = slice
    
    def save_grid(self, filepath):
        nifti = nib.Nifti1Image(self.grid, np.eye(4))
        nib.save(nifti, filepath)
    # ---------------------------------- #--# ----------------------------------
    
    # ---------------------------------- ROI ----------------------------------
    def mask_volumes_roi(self, x_ROI_min, x_ROI_max, y_ROI_min, y_ROI_max, z_ROI_min, z_ROI_max):
        volume_roi = np.zeros_like(self.volume)
        volume_roi[x_ROI_min:x_ROI_max, y_ROI_min:y_ROI_max, z_ROI_min:z_ROI_max] = self.volume[x_ROI_min:x_ROI_max, y_ROI_min:y_ROI_max, z_ROI_min:z_ROI_max]
        self.volume = volume_roi
        
        if self.vessel_volume is not None:
            vessel_volume_roi = np.zeros_like(self.vessel_volume)
            vessel_volume_roi[x_ROI_min:x_ROI_max, y_ROI_min:y_ROI_max, z_ROI_min:z_ROI_max] = self.vessel_volume[x_ROI_min:x_ROI_max, y_ROI_min:y_ROI_max, z_ROI_min:z_ROI_max]
            self.vessel_volume = vessel_volume_roi
            
        if self.brain_volume is not None:
            brain_volume_roi = np.zeros_like(self.brain_volume)
            brain_volume_roi[x_ROI_min:x_ROI_max, y_ROI_min:y_ROI_max, z_ROI_min:z_ROI_max] = self.brain_volume[x_ROI_min:x_ROI_max, y_ROI_min:y_ROI_max, z_ROI_min:z_ROI_max]
            self.brain_volume = brain_volume_roi
    # ---------------------------------- #--# ----------------------------------
    
    def get_random_patches(self, mode=None):
        assert mode in ['vessel', 'empty', None], f'Invalid mode: {mode}'
        self.get_random_vessel_patches() if 'vessel' in mode else self.get_random_empty_patches()
    
    def get_random_vessel_patches(self):
        pass
    
    def get_random_empty_patches(self):
        pass
    
        
def cl_score(v, s):
    return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):
    from skimage.morphology import skeletonize, skeletonize_3d
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)