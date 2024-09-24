from abc import ABC
import os, re

import numpy as np

class Dataset(ABC):
    def __init__(self, datapath):
        self.datapath = datapath
        self.all_paths = None
        self.all_volumes = None
        self.all_vessels = None
        self.weak_vessels = None
        self.all_brains = None
        self.ids = None
    
    def get_len(self):
        return len(self.get_volume_paths())
    
    def get_datapath(self):
        return self.datapath
    
    def get_dataset_paths(self, sort = False):
        self.all_paths = self.getAllFiles(self.datapath, sort=sort) if self.all_paths is None else self.all_paths
        return self.all_paths
    
    def get_ids(self):
        self.all_paths = self.get_dataset_paths()
        self.ids = remove_duplicates([self.get_id_item(item) for item in self.all_paths])
        if self.all_volumes:
            assert len(self.ids) == len(self.all_volumes), f'{len(self.ids)} != {len(self.all_volumes)}'
        elif self.all_brains:
            assert len(self.ids) == len(self.all_brains), f'{len(self.ids)} != {len(self.all_brains)}'
        elif self.all_vessels:
            assert len(self.ids) == len(self.all_vessels), f'{len(self.ids)} != {len(self.all_vessels)}'
        return self.ids

    def get_id_item(self, item):
        id = item.split('/')[-1].split('_')[0]
        assert id.isdigit(), f'Id is not a digit: {id} for item: {item}'
        return str(id)
    
    def filter_paths(self, filtering = None):
        if self.all_paths is None:
            self.all_paths = self.get_dataset_paths()
        
        if filtering is None:
            return self.all_paths
        
        return sorted([item for item in self.all_paths if re.search(filtering, item)])
    
    def get_volume_paths(self, filtering = '_img|_ToF'):
        if self.all_volumes is None:
            self.all_volumes = self.filter_paths(filtering)
        if len(self.all_volumes) == 0:
            print('No volume paths found')
        return self.all_volumes
        
    def get_brain_paths(self, filtering = '_mask'):
        self.all_brains = self.filter_paths(filtering)
        if len(self.all_brains) == 0:
            print('No brain paths found')
        if len(self.all_brains) != len(self.all_volumes):
            print('Number of brain paths does not match number of volume paths')
            if len(self.all_brains) == 2*len(self.all_volumes):
                print('Removing extra brain paths')
                self.all_brains = self.filter_paths(filtering='brain_mask')
        return self.all_brains
    
    def get_label_paths(self, filtering = '_label|_vessel'):
        self.all_vessels = self.filter_paths(filtering)
        
        if len(self.all_vessels) == 0:
            print('No label paths found')
        
        if self.all_volumes is not None:
            if len(self.all_vessels) != len(self.all_volumes):
                print('Number of label paths does not match number of volume paths')
        return self.all_vessels
        
    
    def load_weak_vessels_paths(self, wv_path):
        weak_labels_unfiltered_paths = self.getAllFiles(wv_path, sort=True)
        if self.ids is None:
            self.get_ids()    
        self.weak_vessels = [item for item in weak_labels_unfiltered_paths if self.get_id_item(item) in self.ids]
        return self.weak_vessels
    
    def load_true_vessels_paths(self, tv_path):
        true_labels_unfiltered_paths = self.getAllFiles(tv_path, sort=True)
        
        if self.ids is None:
            self.get_ids()    
        self.true_vessels = [item for item in true_labels_unfiltered_paths 
                             if re.search('_label|_vessel', item) and self.get_id_item(item) in self.ids]
        return self.true_vessels
    
    def load_metadata(self, train_metadata_filepath):
        self.train_metadata = np.load(train_metadata_filepath, allow_pickle=True)
        return self.train_metadata
    
    def getAllFiles(self, dir, result = None, sort = False):
        if result is None:
            result = []
        for entry in os.listdir(dir):
            entrypath = os.path.join(dir, entry)
            if os.path.isdir(entrypath):
                self.getAllFiles(entrypath ,result)
            else:
                result.append(entrypath)
        if sort:
            result = sorted(result)
        return result
    
class GridDataset(Dataset):
    def __init__(self, datapath):
        super().__init__(datapath)
    
    def get_grid_volume_paths(self, filtering = '_grid'):
        self.all_volumes = self.filter_paths(filtering)
        if len(self.all_volumes) == 0:
            print('No grid volume paths found')
        return self.all_volumes 
    
    
class TrainingSet(Dataset):
    def __init__(self, datapath):
        super().__init__(datapath)

class TestSet(Dataset):
    def __init__(self, datapath):
        super().__init__(datapath)
    
class ActiveLearningSet(Dataset):
    def __init__(self, datapath):
        super().__init__(datapath)
        

def remove_duplicates(given_list):
    return list(set(given_list))