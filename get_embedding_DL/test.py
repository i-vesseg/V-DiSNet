import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm

from fastcluster import linkage as fast_linkage
from scipy.cluster.hierarchy import dendrogram


path_data = "/data/falcetta/brain_data"
save_embeddings_path = os.path.join(path_data, f"embeddings_VDISNET") 

dataset_name = 'CAS'

save_embeddings_path_SOTA = os.path.join(save_embeddings_path,'SOTA')
print(f"Save embeddings in {save_embeddings_path_SOTA}")

def determine_num_clusters(data, max_clusters=30, n_jobs=12):
    # Efficiently flatten the data
    print(f"DETERMINE NUM CLUSTERS WITH FASTCLUSTER")
    data_flat = data.reshape(data.shape[0], -1)
    
    # Perform hierarchical clustering using 'ward' method with fastcluster
    linked = fast_linkage(data_flat, method='ward', preserve_input=True)
    
    # Create the dendrogram with no plotting and truncation to max_clusters
    dendro = dendrogram(linked, truncate_mode='lastp', p=max_clusters, no_plot=True)
    
    # Get the number of clusters from the dendrogram
    num_clusters = len(dendro['leaves'])
    
    return num_clusters

def select_representatives(cluster_points, cluster_ids, num_representatives):
    cluster_flat = cluster_points.reshape(cluster_points.shape[0], -1)
    representatives = []
    representatives_idx = []
    
    # Initialize by selecting a random point as the first representative
    initial_idx = np.random.choice(len(cluster_flat))
    representatives.append(cluster_flat[initial_idx])
    representatives_idx.append(cluster_ids[initial_idx])
    remaining_indices = set(range(len(cluster_flat))) - {initial_idx}
    
    while len(representatives) < num_representatives:
        max_min_dist = -1
        next_representative = None
        
        for idx in remaining_indices:
            min_dist = min(np.linalg.norm(cluster_flat[idx] - rep) for rep in representatives)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                next_representative = idx
                
        representatives.append(cluster_flat[next_representative])
        representatives_idx.append(cluster_ids[next_representative])
        remaining_indices.remove(next_representative)
    
    return np.array(representatives).reshape(-1, *cluster_points.shape[1:]), representatives_idx


def compute_pairwise_distances(data):
    return pairwise_distances(data)

# def compute_distance_matrix(data, metric='euclidean'):
#     # Compute the full distance matrix using pairwise_distances
#     return pairwise_distances(data, metric=metric)

def compute_distance_matrix(data, metric='euclidean'):
    return pairwise_distances(data, metric=metric, n_jobs=-1)

def compute_distance_matrix(data, metric='euclidean', chunk_size=100):
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
    
    for i in tqdm(range(0, n_samples, chunk_size)):
        for j in range(0, n_samples, chunk_size):
            i_end = min(i + chunk_size, n_samples)
            j_end = min(j + chunk_size, n_samples)
            distance_matrix[i:i_end, j:j_end] = pairwise_distances(data[i:i_end], data[j:j_end], metric=metric, n_jobs=-1)
    
    return distance_matrix

def ClsMC_RS(data, percentage=0.1, num_clusters=0, chunk_size=100):
    if num_clusters == 0:
        print("STEP 0: Determining num clusters")
        # Step 1: Determine the number of clusters using agglomerative clustering and dendrogram
        num_clusters = determine_num_clusters(data)
    print(f"Number of clusters: {num_clusters}")
    
    # Step 2: Perform clustering with the determined number of clusters
    print("STEP 1: Clustering data")
    data_flat = data.reshape(data.shape[0], -1)
    print(f"Data flat shape: {data_flat.shape}, dtype: {data_flat.dtype}")

    # Compute distance matrix
    print("Computing distance matrix")
    if not os.path.exists(os.path.join(save_embeddings_path_SOTA,f'distance_matrix_CAS_RA.npy')):
        distance_matrix = compute_distance_matrix(data_flat, chunk_size=chunk_size)
        np.save(os.path.join(save_embeddings_path_SOTA,f'distance_matrix_CAS_RA.npy'), distance_matrix)
    else:
        print("Loading distance matrix")
        distance_matrix = np.load(os.path.join(save_embeddings_path_SOTA,f'distance_matrix_CAS_RA.npy'))
        
    print("Performing Agglomerative clustering")
    
    if not os.path.exists(os.path.join(save_embeddings_path_SOTA,f'cluster_labels_CAS_RA.npy')):
        clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='complete')
        cluster_labels = clustering.fit_predict(distance_matrix)
        np.save(os.path.join(save_embeddings_path_SOTA,f'cluster_labels_CAS_RA.npy'), cluster_labels)
        print(f"Different clusters: {set(cluster_labels)}")
    else:
        print("Loading cluster labels")
        cluster_labels = np.load(os.path.join(save_embeddings_path_SOTA,f'cluster_labels_CAS_RA.npy'))
        
    print(f"Different clusters: {set(cluster_labels)}")
    
    # Step 3: Representative selection max coverage sampling
    print("STEP 2: Selecting representatives")
    representatives = []
    representatives_idx = []
    for cluster_idx in range(num_clusters):
        cluster_points = data[cluster_labels == cluster_idx]
        cluster_ids = np.where(cluster_labels == cluster_idx)[0]
        print(f"Cluster {cluster_idx} - Num points: {len(cluster_points)}")
        print(f"Cluster IDs: {cluster_ids}")
        reps, id_reps = select_representatives(cluster_points, cluster_ids, int(len(cluster_points) * percentage))
        representatives.extend(reps)
        representatives_idx.extend(id_reps)
        
        print(f"Adding {len(reps)} representatives (TOTAL: {len(representatives)})")
        print(f"Chosen representatives: {id_reps}")
    
    return np.array(representatives), representatives_idx


def main():
    img_embeddings = np.load(os.path.join(save_embeddings_path_SOTA, f'img_embeddings_CAS_RA.npy'))

    print(f"img_embeddings loaded from {save_embeddings_path_SOTA}")
    print(f'img_embeddings shape: {img_embeddings.shape}') #128*4*4 = 2048

    img_embeddings = img_embeddings.astype(np.float32)
    print(f'img_embeddings dtype: {img_embeddings.dtype}')
    img_embeddings_tsne = np.load(os.path.join(save_embeddings_path_SOTA,f'img_embeddings_tsne_CAS_RA.npy'))
    print(f"TSNE img_embeddings loaded")
    print(f"TSNE img_embeddings shape: {img_embeddings_tsne.shape}")
    
    # take 10% of the data at random
    aaa = np.random.choice(len(img_embeddings_tsne), int(len(img_embeddings_tsne)*0.60), replace=False)
    
    img_embeddings_selected = ClsMC_RS(aaa, percentage=0.1, num_clusters=30, chunk_size=15000)

if __name__ == "__main__":
    main()