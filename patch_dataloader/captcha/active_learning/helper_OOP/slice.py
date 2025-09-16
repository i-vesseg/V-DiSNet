import cv2
import numpy as np 
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans

class MedicalImageSlice():
    def __init__(self, image_data, slice_num=None, brain_data=None, vessel_data=None):
        self.slice = image_data
        self.predicted_slice = None
        self.brain_slice = brain_data
        self.vessel_slice = vessel_data
        self.slice_num = slice_num
        self.slice_shape = self.get_dimensions()
        self.height, self.width = self.slice_shape
    
    def get_slice(self):
        return self.slice
    
    def get_slice_num(self):
        return self.slice_num
    
    def get_dimensions(self):
        return self.slice.shape
    
    # ---------------------------------- ROUGH PRED ----------------------------------    
    def generate_rough_prediction(self, patches_tuple, clustering_method, apply_dilation):
        patches, vessel_patches, brain_patches, patch_positions = patches_tuple
        rough_patches = []
        
        for patch, vessel_patch, brain_patch, position in zip(patches, vessel_patches, brain_patches, patch_positions):
            if 2 in vessel_patch:                         
                # Convert the patch image to a numpy array of float32 data type and flatten it to a 1-D vector
                vectorized = patch.reshape((-1, 1))
                vectorized = np.float32(vectorized)
                
                #Termination criteria
                criteria = (cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                attempts = 10
                cluster_num_K = 3 if 0 in brain_patch else 2
                
                if clustering_method == 'kmeans':
            
                    #Applying K-means clustering for the current img_slice") ==> return compactness,label,center
                    # https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
                    #_, label, center = cv2.kmeans(vectorized, cluster_num_K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
                    kmeans = KMeans(n_clusters=cluster_num_K, random_state=0, n_init=10).fit(vectorized)
                    label = kmeans.labels_
                    center = kmeans.cluster_centers_
                    
                    res = center[label.flatten()]
                    rough_patch = res.reshape((patch.shape))
                    
                    rough_patch[rough_patch != np.max(center)] = 0 # Assign 0 to all non-maxima values of the cluster center
                    rough_patch[rough_patch == np.max(center)] = 1 # Assign 1 to the max value of the cluster center
                    
                    rough_patch = rough_patch.astype('int8')
                    
                    # Apply connected component analysis to the thresholded image to identify individual objects.
                    num_labels, labels_im = cv2.connectedComponents(rough_patch)
                    # Apply a threshold to the resulting image, such that only pixels with the highest intensity in each cluster are retained.
                    threshold = 100 if 0 in brain_patch else 350
                    threshold = 350 if 0 in brain_patch and 2 in vessel_patch else threshold
                    
                    if (num_labels < 5) and (np.count_nonzero(rough_patch) < threshold) :
                        if apply_dilation:
                            # Apply dilation to the segmented image
                            kernel = np.ones((2, 2), np.uint8)  # Define the dilation kernel size and shape
                            rough_patch = cv2.dilate(rough_patch.astype(np.uint8), kernel, iterations=1)
                        
                        rough_patches.append(rough_patch) 
                    else:
                        rough_patches.append(np.zeros(patch.shape))
                        # The whole patch is considered as background
                        
                        
                elif clustering_method == 'gmm':
                    pixels = patch[patch != 0]  # Extract non-zero pixels from the image

                    vectorized = pixels.reshape((-1, 1))  # Reshape pixels into a 1D vector
                    vectorized = np.float32(vectorized)

                    n_components = 4 if 0 in brain_patch else 2
                    n_components = 2 if 2 in vessel_patch else n_components

                    gmm_model_tied = GMM(n_components=n_components, covariance_type='tied').fit(vectorized)

                    # Extract the cluster centers, labels and the resulting image
                    center_tied = gmm_model_tied.means_
                    label_tied = gmm_model_tied.predict(vectorized).reshape(-1, 1)
                    res_tied = center_tied[label_tied.flatten()]

                    result_image_tied = res_tied
                    
                    result_image_tied[result_image_tied != np.max(center_tied)] = 0  # Set non-maximum values to 0
                    result_image_tied[result_image_tied == np.max(center_tied)] = 1  # Set maximum value to 1
                    
                    rough_patch_gmm = np.zeros(patch.shape)
                    pos = np.where(patch != 0)  # Find positions where img_patch is non-zero
                    rough_patch_gmm[pos[0], pos[1]] = result_image_tied.reshape(len(patch[patch != 0])) # Assign result_image_tied values to corresponding positions in b
                    
                    rough_patch_gmm = rough_patch_gmm.astype('int8')
                    num_labels, labels_im = cv2.connectedComponents(rough_patch_gmm)
                    
                    threshold = 100 if 0 in brain_patch else 350
                    threshold = 350 if 0 in vessel_patch else threshold
                         
                    if (num_labels < 5) and (np.count_nonzero(rough_patch_gmm) < threshold):
                        if apply_dilation:
                            kernel = np.ones((3, 3), np.uint8)
                            rough_patch_gmm = cv2.dilate(rough_patch_gmm.astype(np.uint8), kernel, iterations=1)
                            
                        rough_patches.append(rough_patch_gmm)      
                else:
                    raise Exception(f'Unknown clustering method: ({clustering_method})')
            
            else:  
                # The patch is black: just empty (0) pixels 
                rough_patches.append(np.zeros(patch.shape))
                
        
        
        return rough_patches
            
    # --------------------------------- # -------- # ----------------------------------
    
    def get_consecutive_patches(self, patch_size, overlap_x=0.0, overlap_y=0.0, get_mask_patches=False):
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        #print("TODO: AAAAAAAAAAAAAAAAAAAAAA MODIFY AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA: Take consecutive patches just where there is brain!!!!")
        
        # Calculate the step size based on the overlap ratios
        assert overlap_x < 1 and overlap_y < 1 and overlap_x >= 0 and overlap_y >= 0, f"Invalid overlap ratio [0, 1), got {overlap_x} and {overlap_y}"
        step_size_y = int(patch_height * (1 - overlap_y))
        step_size_x = int(patch_width * (1 - overlap_x))
        
        #print(f" - Patch size: {patch_size}, height: {patch_height}, width: {patch_width}")
        #print(f" - Step size: {step_size_y}x{step_size_x}")
        #print(f" - {1- overlap_y} overlap in y, {1-overlap_x} overlap in x")
        
        # Initialize empty lists to store patches and their positions
        patches = []
        vessel_patches = []
        patch_positions = []
        
        if get_mask_patches:
            mask_patches = []

        # Iterate through the slice in a sliding-window fashion
        #print("Extracting patches...")
        
        for start_y in range(0, self.height, step_size_y):
            for start_x in range(0, self.width, step_size_x):
                end_y = start_y + patch_height
                end_x = start_x + patch_width
                
                if end_y > self.height:
                    end_y = self.height
                    start_y = self.height - patch_height
                
                if end_x > self.width:
                    end_x = self.width
                    start_x = self.width - patch_width

                if get_mask_patches:
                    mask_patch = self.brain_slice[start_y:end_y, start_x:end_x] if self.brain_slice is not None else None
                    # check if is empty (all zeros) or not
                    if np.count_nonzero(mask_patch) == 0:
                        #print(f"Empty patch at ({start_y}:{end_y}, {start_x}:{end_x})")
                        continue
                    #else:
                        #print(f"Non-empty patch at ({start_y}:{end_y}, {start_x}:{end_x})")
                        
                    mask_patches.append(mask_patch)
                    
                    
                # Extract the patch and store it in the lists
                patch = self.slice[start_y:end_y, start_x:end_x]
                patches.append(patch)
                
                vessel_patch= self.vessel_slice[start_y:end_y, start_x:end_x] if self.vessel_slice is not None else None
                vessel_patches.append(vessel_patch)
                
                patch_positions.append((start_y, end_y, start_x, end_x))
         
        if get_mask_patches:
            return patches, vessel_patches, mask_patches, patch_positions
        
        return patches, vessel_patches, patch_positions


    def reconstruct_slice(self, modified_patches, patch_positions, method=None, threshold = 0.5):
        reconstructed_image = np.zeros_like(self.slice)
        
        if method: 
            patch_counts = np.zeros_like(self.slice, dtype=int)

        for i, position in enumerate(patch_positions):
            start_y, end_y, start_x, end_x = position
            
            if method is None:
                reconstructed_image[start_y:end_y, start_x:end_x] = modified_patches[i]
            else:
                reconstructed_image[start_y:end_y, start_x:end_x] += modified_patches[i]
                patch_counts[start_y:end_y, start_x:end_x] += 1

    
        if method:
            patch_counts[patch_counts == 0] = 1  # Avoid division by zero
            reconstructed_image /= patch_counts # Combine patches by averaging in the overlapping regions
            reconstructed_image = np.where(reconstructed_image > threshold, 1, 0)  # Threshold the reconstructed image

        self.predicted_slice = reconstructed_image
        return self.predicted_slice

    def get_reconstructed_slice(self):
        return self.predicted_slice
    
    def get_grid(self):
        return self.grid
    
    def create_grid(self, vessel_patches, patch_positions):
        
        self.grid = np.zeros_like(self.slice)
        
        for i, (position, vessel_patch) in enumerate(zip(patch_positions, vessel_patches)):
            unique_label = np.unique(vessel_patch).astype(int)

            assert (len(unique_label) == 2 and 1 in unique_label) or (len(unique_label) == 1 and 0 in unique_label), f"Invalid unique label: {unique_label} for slice {self.slice_num}, patch {i} - {position}"
            
            start_x, end_x, start_y, end_y = position
            
            if 1 in unique_label:
                self.grid[start_x: end_x, start_y: end_y] = 2
        
