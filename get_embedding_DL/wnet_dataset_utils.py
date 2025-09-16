import numpy as np
from torch.utils.data import Dataset
import torch

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
    