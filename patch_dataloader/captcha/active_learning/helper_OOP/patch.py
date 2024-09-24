import os
from matplotlib import pyplot as plt
import numpy as np
import torch

class MedicalImagePatch:
    def __init__(self, patch, vessel_patch=None, coords=None, patch_size=None, device='cpu'):
        
        self.patch = patch
        self.vessel_patch = vessel_patch
        self.true_class = 1 if 1. in (np.unique(self.vessel_patch)) else 0
        #if self.true_class == 1:
            #print("AAAA")
        self.torch_patch, self.torch_vessel_patch = None, None
        self.predicted_vessel_patch = None
        if self.patch.ndim == 2:
            self.x_start, self.x_end, self.y_start, self.y_end = coords
            self.width = self.x_end - self.x_start
            self.height = self.y_end - self.y_start
            assert self.width == self.height, f"Patch width {self.width} != patch height {self.height}"
            self.coords = coords
        else:
            self.width = (0, 0)
            self.height = (0, 0)
            self.coords = (0, 0, 0, 0)
        self.get_coords()
        self.patch_size = patch_size if patch_size is not None else self.width
        self.device = device
        
    def get_area(self):
        return self.width * self.height
    
    def get_prediction(self):
        return self.predicted_vessel_patch.to('cpu').numpy()
    
    def get_coords(self):
        if self.coords == (0, 0, 0, 0):
            #print("No coordinates are available (patch is not a 2D image)")
            pass
        return self.coords
    
    
    def normalize(self, metadata):
        
        try:
            mean, std = metadata['mean'], metadata['std'] 
        except:
            mean, std = metadata
            
        mean = np.array(mean, dtype=float)
        std = np.array(std + 1e-7, dtype=float) if std == 0 else np.array(std, dtype=float)
        
        self.patch = (self.patch - mean) / std
        
        self.patch = (self.patch - self.patch.min()) / (self.patch.max() - self.patch.min())
        return self.patch
    
    def get_torch_patch(self):
        if self.patch.ndim == 2:
            self.torch_patch = torch.from_numpy(self.patch).to(torch.float32).view(1, 1, self.height, self.width).to(self.device)
            self.torch_vessel_patch = torch.from_numpy(self.vessel_patch).to(torch.float32).view(1, 1, self.height, self.width).to(self.device) if self.vessel_patch is not None else None
        elif self.patch.ndim == 3:
            self.torch_patch = torch.from_numpy(self.patch).to(torch.float32).view(self.patch.shape[0], 1, self.patch.shape[1], self.patch.shape[2]).to(self.device)
            self.torch_vessel_patch = torch.from_numpy(self.vessel_patch).to(torch.float32).view(self.vessel_patch.shape[0], 1, self.vessel_patch.shape[1], self.vessel_patch.shape[2]).to(self.device) if self.vessel_patch is not None else None
        return self.torch_patch, self.torch_vessel_patch
    
    def predict(self, model=None, test_time_augmentation=False, tta_transforms=None):
        if model is None:
            raise ValueError("No model is specified")
        overall_tta_uncertainty = None
        self.torch_patch, self.torch_vessel_patch = self.get_torch_patch()
        with torch.no_grad():
            if test_time_augmentation:
                tta_predictions = []
                if tta_transforms is None:
                    raise ValueError("No test time augmentation transform is specified")
                
                for i, tta_transformer in enumerate(tta_transforms):
                    
                    augmented_patch = tta_transformer.augment_image(self.torch_patch)
                    ###################################################################
                    #save_fig(augmented_patch, self.torch_patch, name = f't{1}_augmented_patch')
                    ###################################################################
                    
                    tta_prediction_aug = model(augmented_patch)
                    tta_prediction = tta_transformer.deaugment_mask(tta_prediction_aug)
                    
                    ###################################################################
                    #save_fig(tta_prediction_aug,tta_prediction, f't{i}_predicted_patch')
                    ###################################################################
                    
                    tta_predictions.append(tta_prediction)
                
                tta_predictions = torch.stack(tta_predictions)
                
                # TODO: COMPUTE UNCERTAINTY
                tta_uncertainty = torch.var(tta_predictions, dim=0)
                overall_tta_uncertainty = torch.mean(tta_uncertainty)
                
                self.predicted_vessel_patch = torch.mean(tta_predictions, dim=0)
                    
            else:
                self.predicted_vessel_patch = model(self.torch_patch)
                
        return self.predicted_vessel_patch, overall_tta_uncertainty
    
    def predict_patch(self, model=None, PNET_selection=True):
        if model is None:
            raise ValueError("No model is specified")
        
        self.torch_patch, self.torch_vessel_patch = self.get_torch_patch()
        
        if torch.sum(self.torch_vessel_patch) == 0 and PNET_selection:
            self.predicted_vessel_patch = torch.zeros(1, 1, self.height, self.width)
        else:
            with torch.no_grad():
                self.predicted_vessel_patch = model(self.torch_patch)
                    
        return self.predicted_vessel_patch
    
    def MCDO_predict(self, model=None, test_time_augmentation=False, tta_transforms=None, MC_dropout_runs=1):
        if model is None:
            raise ValueError("No model is specified")
        
        MC_predictions = []
        self.torch_patch = self.get_torch_patch()
        for _ in range(MC_dropout_runs):
            MC_prediction, overall_tta_uncertainty = self.predict(model=model, test_time_augmentation=test_time_augmentation, tta_transforms=tta_transforms)
            MC_predictions.append(MC_prediction)
        
        MC_predictions = torch.stack(MC_predictions)
        
        # TODO: COMPUTE UNCERTAINTY
        MC_uncertainty = torch.var(MC_predictions, dim=0)
        overall_MC_uncertainty = torch.mean(MC_uncertainty)
        
        self.predicted_vessel_patch = torch.mean(MC_predictions, dim=0)
        
        overall_MC_uncertainty = overall_MC_uncertainty.cpu().numpy().item()
        overall_tta_uncertainty = overall_tta_uncertainty.cpu().numpy().item() if overall_tta_uncertainty is not None else None
        
        return overall_MC_uncertainty, overall_tta_uncertainty
    
    def compute_metric_acc(self, metric=None):
        if self.predicted_vessel_patch is None:
            print("No prediction is available")
            return None
        
        if metric is None:
            raise ValueError("No metric is specified")
        
        self.predicted_vessel_patch = torch.argmax(self.predicted_vessel_patch).unsqueeze(0)
        true_class_torch = torch.tensor(self.true_class).to(self.device).unsqueeze(0)
        
        assert true_class_torch.shape == self.predicted_vessel_patch.shape, f"True class shape {true_class_torch.shape} != predicted shape {self.predicted_vessel_patch.shape}"
        
        return metric(self.predicted_vessel_patch, true_class_torch).cpu().numpy().item()
        
    def compute_metric(self, metric=None):
        if self.predicted_vessel_patch is None:
            print("No prediction is available")
            return None
        
        if metric is None:
            raise ValueError("No metric is specified")
        
        self.predicted_vessel_patch =  self.predicted_vessel_patch.view(self.patch_size, self.patch_size).to(self.device)

        torch_vessel_patch = torch.from_numpy(self.vessel_patch).to(self.device).view(1,1,self.patch_size, self.patch_size).to(torch.int32)
        torch_prediction = self.predicted_vessel_patch.view(1,1,self.patch_size, self.patch_size)
        
        assert torch_prediction.shape == torch_vessel_patch.shape, f"Predicted shape {torch_prediction.shape} != gt shape {torch_vessel_patch.shape}"
        
        return metric(torch_prediction, torch_vessel_patch)
        
    def prediction_argmax(self):
        self.predicted_vessel_patch = torch.argmax(self.predicted_vessel_patch, dim=0, keepdim=True)
    
    def reshape_and_threhsold_prediction(self, threshold=0.5, generate_patch_from_class_pred=False):
        
        if generate_patch_from_class_pred:
            self.predicted_vessel_patch = torch.zeros(self.patch_size, self.patch_size) if self.predicted_vessel_patch == 0 else torch.ones(self.patch_size, self.patch_size)
            self.predicted_vessel_patch.float()       
        else:
            if self.predicted_vessel_patch.shape[0] == 1:
                self.predicted_vessel_patch = self.predicted_vessel_patch.view(self.patch_size, self.patch_size)
            elif self.predicted_vessel_patch.shape[0] > 1:
                self.predicted_vessel_patch = self.predicted_vessel_patch.view(self.predicted_vessel_patch.shape[0], self.patch_size, self.patch_size)
                
            self.predicted_vessel_patch = (self.predicted_vessel_patch > threshold).float()
        
    
    
    def get_entropy(self, method='entropy', n_classes=1):
        if n_classes == 1:
            reshaped_pred = self.predicted_vessel_patch.view(1, self.patch_size, self.patch_size) # Resize to remove the batch dimension [1, 1, height, width] -> [1, height, width]

            # Assign probabilities to the tensor
            transformed_pred = torch.zeros(2, self.patch_size, self.patch_size) # Create an empty tensor with 2 channels to encode the probability distribution [1, 2, height, width]
            transformed_pred[0, :, :] = 1 - reshaped_pred # Background probability (1 - Vessel probability)
            transformed_pred[1, :, :] = reshaped_pred # Vessel probability 

        #print(f"The prediction has shape {transformed_pred.shape}") # [2, height, width]

        entropy_uncertainty = compute_uncertainty(transformed_pred, method=method) # compute entropy along the first dimension (n_classes)
        return entropy_uncertainty
    
def compute_uncertainty(prediction, method='entropy'):
    # prediction: [n_classes, height, width]
    
    assert method in ['entropy', 'max_prob'], f'Invalid uncertainty method: {method}'
    assert prediction.ndim == 3, f'Invalid prediction dimension: {prediction.ndim}, should be 3'
    assert prediction.shape[0] > 1, f'Invalid prediction shape: {prediction.shape}, should be [n_classes, height, width]'
    
    if method == 'entropy': # Compute entropy of the prediction per pixel 
        per_pixel_uncertainty = - torch.sum(prediction * torch.log(prediction + 1e-6), dim=0) # [height, width]
    
    elif method == 'max_prob': # Consider the maximum probability of the prediction per pixel
        per_pixel_uncertainty = 1 - torch.max(prediction, dim=0)[0] # [height, width]
        
    uncertainty = torch.mean(per_pixel_uncertainty) # Compute the (scalar) uncertainty by averaging the uncertainty per pixel
    uncertainty = uncertainty.cpu().numpy().item()
    return uncertainty # The higher the uncertainty, the more information you can gain by inserting this patch to the training set
        

# TODO: Entropy, overall_MC_uncertainty, overall_tta_uncertainty, random ==> For each patch, write in a file (heapq.heappush(SCORESSS, (SCORE, (patient_filename, slice_num, x_patch_num, y_patch_num))))

def save_fig(img_1, img_2, name):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_1[0, 0, :, :].cpu().numpy())
    ax[1].imshow(img_2[0, 0, :, :].cpu().numpy())
    name = get_new_name(name)
    
    plt.savefig(name)

def get_new_name(name):
    if os.path.exists(name):
        if name[0].isdigit():
            num, name = name.split('_')[0], '_'.join(name.split('_')[1:])
            id = f"{int(num) + 1}_"
        else:
            id = f"{1}_"
            
        name =  id + name
        name = get_new_name(name)
    return name

