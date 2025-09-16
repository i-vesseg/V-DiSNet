


import os
import numpy as np
from sklearn.metrics import pairwise_distances

# ----------------------------------------------------------------------------------------------- #
def random_sampling(*embedding_lists, n_size=1000, random_seed=42):
    # Random sampling
    print(f"Random seed: {random_seed}")
    np.random.seed(random_seed)
    embedding_arrays = [np.array(embedding_list) for embedding_list in embedding_lists]
    # Assuming all embedding lists have the same length
    num_samples = embedding_arrays[0].shape[0]
    
    assert len(set(len(embedding_list) for embedding_list in embedding_lists)) == 1, "All embedding lists should have the same length."
    
    if n_size<1: # percentage of num_samples
        print(f"WARNING: n_size parameter is <1 ({n_size})")
        print(f"Taking the {n_size*100}% of the num_samples ({num_samples}) ==> {int(n_size * num_samples)}")
        n_size = int(n_size * num_samples)
        
    # Randomly select indices
    random_idx = np.random.choice(num_samples, n_size, replace=False)
    # Perform random sampling for each embedding list
    random_lists = []
    for embedding_array in embedding_arrays:
        random_list = embedding_array[random_idx].tolist()
        random_lists.append(random_list)
    random_lists = [np.array(random_list) for random_list in random_lists]
    return random_lists, random_idx
# ----------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------- #
def random_class_sampling(*embedding_lists, class_array, n_size=1000, random_seed=42):
    # Random sampling
    print(f"Random seed: {random_seed}")
    np.random.seed(random_seed)
    embedding_arrays = [np.array(embedding_list) for embedding_list in embedding_lists]
    
    # Assuming all embedding lists have the same length
    assert len(set(len(embedding_list) for embedding_list in embedding_lists)) == 1, "All embedding lists should have the same length."
    
    # Group indices by class
    class_indices = {}
    class_count = {}
    for i, class_label in enumerate(class_array):
        if class_label not in class_indices:
            class_indices[class_label] = []
            class_count[class_label] = 0
        class_indices[class_label].append(i)
        class_count[class_label]+=1
    
    # Randomly select indices for each class
    random_idx = []
    if n_size<1:
        # Calculate the number of samples per class (SAME PERCENTAGE FOR EACH CLUSTER)
        print(f"Warning: n_size<1 ({n_size})")
        print(f"Taking {n_size*100}% of samples for each cluster")
        
        if set(class_indices.keys()) == set(class_count.keys()):
            # print("Keys are the same in both dictionaries")
            for key in class_indices.keys():
                indices = class_indices[key]
                count_c = class_count[key]
                #print(f"Key: {key}, Value in dict1: {indices}, Value in dict2: {count_c}")
                samples_per_class = int(count_c * n_size)
                print(f"Taking {samples_per_class} for cluster {key}")
                random_idx.extend(np.random.choice(indices, samples_per_class, replace=False))
        else:
            print("Keys are not the same in both dictionaries")
            assert()
        
    
    else:
        # Calculate the number of samples per class (SAME NUMBER FOR EACH CLUSTER
        samples_per_class = n_size // len(class_indices)
        print(f"Taking {samples_per_class} samples per class")
        for indices in class_indices.values():
           random_idx.extend(np.random.choice(indices, samples_per_class, replace=(len(indices) < samples_per_class)))
            
    
    # Perform random sampling for each embedding list
    random_lists = []
    for embedding_array in embedding_arrays:
        random_list = embedding_array[random_idx].tolist()
        random_lists.append(random_list)
    
    random_lists = [np.array(random_list) for random_list in random_lists]
    return random_lists, random_idx
# ----------------------------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------------------------- #
def farthest_point_sampling(X, num_points, random_seed):
    """
    Perform farthest point sampling to select a subset of points from X.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
        The input data points.
    - num_points: int
        The number of points to select.

    Returns:
    - selected_indices: array-like, shape (num_points,)
        Indices of the selected points.
    """

    # Initialize an empty list to store selected point indices
    selected_indices = []

    # Choose a random point to start with
    np.random.seed(random_seed)
    initial_index = np.random.choice(X.shape[0])
    selected_indices.append(initial_index)

    # Compute pairwise distances from the selected point to all other points
    distances = pairwise_distances(X, [X[initial_index]])

    # Iterate until we select num_points
    while len(selected_indices) < num_points:
        # Find the point farthest from the selected set
        farthest_index = np.argmax(distances)

        # Update the distances array by choosing the minimum between the existing
        # distances and the distances from the newly selected point
        distances = np.minimum(distances, pairwise_distances(X, [X[farthest_index]]))

        # Add the farthest point to the selected set
        selected_indices.append(farthest_index)

    return selected_indices
# ----------------------------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------------------------- #
def sample_within_class(coordinates, classes, n_size, *arrays, random_seed, single_cluster=False):
    """
    Sample m/p samples from each class using farthest point sampling.
    """
    print(f"Random seed: {random_seed}")
    if single_cluster:
        # Consider the whole dataset as a single class
        classes = np.zeros_like(classes)
    
    
    unique_classes = np.unique(classes)
    sampled_indices = []
    if n_size<1:
        # Calculate the number of samples per class (SAME PERCENTAGE FOR EACH CLUSTER)
        print(f"Warning: n_size<1 ({n_size})")
        print(f"Taking {n_size*100}% of samples for each cluster")
        
    for c in unique_classes:
        # Find indices of samples belonging to class c
        class_indices = np.where(classes == c)[0]
        num_samples_class = len(class_indices)
        
        if n_size <1:
            samples_to_select = int(n_size*num_samples_class)
        else:
            samples_to_select = min(n_size // len(unique_classes), num_samples_class)

        # Perform farthest point sampling within the class
        sampled_indices_class = farthest_point_sampling(coordinates[class_indices], samples_to_select, random_seed=random_seed)
        print(f"Class {c}: {len(sampled_indices_class)} samples selected")
        sampled_indices.extend(class_indices[sampled_indices_class])

    # Apply the sampled indices to other arrays
    sampled_arrays = [array[sampled_indices] for array in (coordinates, classes) + arrays]

    return sampled_arrays 

# ----------------------------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------------------------- #
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

def calculate_density(features, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(features)
    distances, _ = nbrs.kneighbors(features)
    densities = np.sum(distances, axis=1)
    return densities

def select_representative_samples(densities, num_samples):
    print(f"Selecting {num_samples} samples from {len(densities)} total samples")
    indices = np.argsort(densities)[:num_samples]
    return indices

def one_shot_AET(features, num_samples, k=5):
    
    if num_samples < 1:
        print(f"WARNING: num_samples parameter is <1 ({num_samples})")
        print(f"Taking the {num_samples*100}% of the num_samples ({len(features)}) ==> {int(num_samples * len(features))}")
        num_samples = int(num_samples * len(features))

    # Normalize features
    features = normalize(features)
    
    # Step 2: Calculate density of samples in feature space
    densities = calculate_density(features, k)
    
    # Step 3: Select samples with higher local density
    selected_indices = select_representative_samples(densities, num_samples)
    
    return selected_indices

def filter_by_index(idx, *arrays):
    return (arr[idx] for arr in arrays)

def AET_sampling(features, *embedding_lists, n_size=1000, random_seed=42):
    np.random.seed(random_seed)
    selected_indices = one_shot_AET(features, n_size)
    return filter_by_index(selected_indices, *embedding_lists)
# ----------------------------------------------------------------------------------------------- #

def get_samples(save_embeddings_path, dataset_name, n_size, take, current_seed, original_dataset='CAS'):

    X_test = np.load(os.path.join(save_embeddings_path,f'X_test_{original_dataset}_{take}.npy'))
    X_test_mask = np.load(os.path.join(save_embeddings_path,f'X_test_mask_{original_dataset}_{take}.npy'))

    X_test_empty = np.load(os.path.join(save_embeddings_path,f'X_test_empty_{original_dataset}_{take}.npy'))
    X_test_mask_empty = np.load(os.path.join(save_embeddings_path,f'X_test_empty_mask_{original_dataset}_{take}.npy'))
    
    X_test_tot = np.concatenate((X_test, X_test_empty), axis=0)
    X_test_mask_tot = np.concatenate((X_test_mask, X_test_mask_empty), axis=0)
    
    # Overwrite the test set with the tot set
    X_test = X_test_tot
    X_test_mask = X_test_mask_tot
    
    if dataset_name == 'all':
        return X_test, X_test_mask
    
    # Get img embeddings
    print(f'Loading img embeddings from {save_embeddings_path}:')
    img_embeddings = np.load(os.path.join(save_embeddings_path, f'img_embeddings_{original_dataset}_{take}.npy'))
    print(f'img_embeddings shape: {img_embeddings.shape}')
    
    img_embeddings_tsne = np.load(os.path.join(save_embeddings_path,f'img_embeddings_tsne_{original_dataset}_{take}.npy'))
    print(f"TSNE img_embeddings shape: {img_embeddings_tsne.shape}")
    
    img_cluster_labels_2d = np.load(os.path.join(save_embeddings_path,f'img_cluster_labels_2d_{original_dataset}_{take}.npy'))
    print(f"TSNE Cluster labels shape: {img_cluster_labels_2d.shape}")
    
    img_cluster_labels = np.load(os.path.join(save_embeddings_path,f'img_cluster_labels_{original_dataset}_{take}.npy'))
    print(f"Latent Cluster labels shape: {img_cluster_labels.shape}")
    
    if dataset_name == 'R': # RANDOM SAMPLING (R-W)
        selected_lists_r, selected_indices_r = random_sampling(X_test, X_test_mask, img_embeddings, img_embeddings_tsne ,img_cluster_labels, img_cluster_labels_2d, n_size=n_size, random_seed=current_seed)
        
        X_test_r, X_test_mask_r, img_embeddings_r,img_embeddings_tsne_r, img_cluster_labels_r, img_cluster_labels_2d_r = selected_lists_r
        return X_test_r, X_test_mask_r
    
    elif dataset_name == 'C': # RANDOM SAMPLING WITHIN CLASS (LATENT) (R-C)
        selected_lists_c, selected_indices_c = random_class_sampling(X_test, X_test_mask, img_embeddings, img_embeddings_tsne ,img_cluster_labels, img_cluster_labels_2d,
                                                                        class_array=img_cluster_labels, n_size=n_size, random_seed=current_seed)
        
        X_test_c, X_test_mask_c, img_embeddings_c,img_embeddings_tsne_c, img_cluster_labels_c, img_cluster_labels_2d_c = selected_lists_c
        return X_test_c, X_test_mask_c
    
    elif dataset_name == 'CF': # FAR POINT SAMPLING WITHIN CLASS (TSNE) (FT-C)
        sampled_arrays_cf = sample_within_class(img_embeddings_tsne, img_cluster_labels, n_size, X_test_mask, X_test, img_embeddings, img_cluster_labels_2d, random_seed=current_seed)
        
        img_embeddings_tsne_cf, img_cluster_labels_cf, X_test_mask_cf, X_test_cf, img_embeddings_cf, img_cluster_labels_2d_cf = sampled_arrays_cf
        return X_test_cf, X_test_mask_cf
    
    elif dataset_name == 'CFX': # FAR POINT SAMPLING WITHIN CLASS (LATENT) (FL-C)
        sampled_arrays_cfx = sample_within_class(img_embeddings, img_cluster_labels, n_size, X_test_mask, X_test, img_embeddings_tsne, img_cluster_labels_2d, random_seed=current_seed)
        
        img_embeddings_cfx, img_cluster_labels_cfx, X_test_mask_cfx, X_test_cfx, img_embeddings_tsne_cfx, img_cluster_labels_2d_cfx = sampled_arrays_cfx
        return X_test_cfx, X_test_mask_cfx

        
def get_SOTA_samples(save_embeddings_path, dataset_name, n_size, take, current_seed, original_dataset='CAS'):

    X_test = np.load(os.path.join(save_embeddings_path,f'X_test_{original_dataset}_{take}.npy'))
    X_test_mask = np.load(os.path.join(save_embeddings_path,f'X_test_mask_{original_dataset}_{take}.npy'))

    X_test_empty = np.load(os.path.join(save_embeddings_path,f'X_test_empty_{original_dataset}_{take}.npy'))
    X_test_mask_empty = np.load(os.path.join(save_embeddings_path,f'X_test_empty_mask_{original_dataset}_{take}.npy'))
    
    X_test_tot = np.concatenate((X_test, X_test_empty), axis=0)
    X_test_mask_tot = np.concatenate((X_test_mask, X_test_mask_empty), axis=0)
    
    # Overwrite the test set with the tot set
    X_test = X_test_tot
    X_test_mask = X_test_mask_tot
    
    # Get img embeddings
    SOTA_name = dataset_name.split('_')[-1]
    SOTA_path = os.path.join(save_embeddings_path, f'SOTA')
    
    print(f'Loading img embeddings from {SOTA_path}:')
    img_embeddings = np.load(os.path.join(SOTA_path, f'img_embeddings_{original_dataset}_{SOTA_name}.npy'))
    print(f'img_embeddings shape: {img_embeddings.shape}')
    
    img_embeddings_tsne = np.load(os.path.join(SOTA_path,f'img_embeddings_tsne_{original_dataset}_{SOTA_name}.npy'))
    print(f"TSNE img_embeddings shape: {img_embeddings_tsne.shape}")
    
    img_cluster_labels_2d = np.load(os.path.join(SOTA_path,f'img_cluster_labels_2d_{original_dataset}_{SOTA_name}.npy'))
    print(f"TSNE Cluster labels shape: {img_cluster_labels_2d.shape}")
    
    img_cluster_labels = np.load(os.path.join(SOTA_path,f'img_cluster_labels_{original_dataset}_{SOTA_name}.npy'))
    print(f"Latent Cluster labels shape: {img_cluster_labels.shape}")
    
    
    if dataset_name == 'SOTA_CA' or dataset_name == 'SOTA_RA':
        sampled_arrays_cf = sample_within_class(img_embeddings_tsne, img_cluster_labels_2d, n_size, X_test_mask, X_test, img_embeddings, img_cluster_labels, random_seed=current_seed)
        img_embeddings_tsne_cf, img_cluster_labels_2d_cf, X_test_mask_cf, X_test_cf, img_embeddings_cf, img_cluster_labels_cf = sampled_arrays_cf
        return X_test_cf, X_test_mask_cf
    elif dataset_name == 'SOTA_AET':
        X_test_aet, X_test_mask_aet, img_embeddings_aet, img_embeddings_tsne_aet, img_cluster_labels_aet, img_cluster_labels_2d_aet = AET_sampling(img_embeddings, X_test, X_test_mask, img_embeddings, img_embeddings_tsne, img_cluster_labels, img_cluster_labels_2d, n_size=n_size, random_seed=current_seed)
        return X_test_aet, X_test_mask_aet
