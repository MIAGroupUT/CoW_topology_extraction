from monai.transforms import Transform
from torch.nn.functional import grid_sample
import torch
import numpy as np

class GetStartEndPoints(Transform):
    """
    Return the start- and endpoints of each artery in the CoW as a dictionary
    """
    def __init__(self, BoI_key):
        self.cow_dict = {'AcoA': (0, 1), 'ACA_A1_R': (1, 2), 'ACA_A1_L': (0, 3)
           , 'PcoA_R': (5, 7), 'PcoA_L': (4, 6), 'PCA_P1_R': (8,7), 
            'PCA_P1_L': (8,6)}
        self.BoI_key = BoI_key
        
    def __call__(self, data):
        points_dict = {}
        for vessel in list(self.cow_dict):
            inds = self.cow_dict[vessel]
            points_dict[vessel] = (data[self.BoI_key]['grid'][inds[0]],
                                   data[self.BoI_key]['grid'][inds[1]])
        data['seedpoints'] = points_dict
        return data
    

class InterpolateImage(Transform):
    """
    Interpolate the image based on the SIRE grid
    """
    
    def __init__(self, img_key, grid_key, mode='bilinear'):
        self.img_key = img_key
        self.grid_key = grid_key
        self.mode = mode
#         self.vectorfield_key = vectorfield_key
        
    def __call__(self, data):
        grid = ((2 * data[self.grid_key]) / torch.flip(torch.tensor(data[self.img_key].shape), dims=[0])) - 1
        # img with [x,y,z] coordinates (same as vector field!)
        data[f'int_{self.img_key}'] = grid_sample(data[self.img_key].unsqueeze(0).unsqueeze(0),
                             grid.view(1,1,1,-1,3), mode=self.mode).view(data[f'{self.grid_key}_meta']['shape'])
        return data
    
    
def get_cos_similarity(vectorfield, mask_defaults=True): # very non-optimized
    activations = vectorfield[7,:,:,:]
    cos_sim = np.zeros_like(activations)
    D1, D2 = vectorfield[:3,:,:,:], vectorfield[3:6,:,:,:]
    for x in range(cos_sim.shape[0]):
        for y in range(cos_sim.shape[1]):
            for z in range(cos_sim.shape[2]):
                # all cos_sims compared to neighbouring vertices
                neighbors = get_voxel_neighbors(np.transpose(D1,[1,2,3,0]).shape, x, y, z)
                sims = []
                for neighbor in neighbors:
                    # get cosine similarity between D1 from neighbour
                    cossim1 = np.dot(D1[:,x,y,z], 
                                     D1[:,neighbor[0], neighbor[1], neighbor[2]])
                    cossim2 = np.dot(D1[:,x,y,z], 
                                     D2[:,neighbor[0], neighbor[1], neighbor[2]])
                    if cossim1 > cossim2: #cossim D2
                        sims.append(np.dot(D2[:,x,y,z], 
                                           D2[:,neighbor[0], neighbor[1], neighbor[2]]))
                        sims.append(cossim1)
                    else:
                        sims.append(np.dot(D2[:,x,y,z], 
                                           D1[:,neighbor[0], neighbor[1], neighbor[2]]))
                        sims.append(cossim2)
                cos_sim[x,y,z] = np.mean(np.stack(sims))
    if mask_defaults:
        inds = np.where(activations == 0)
        # set cos_sim to -1 where the defaults are set
        cos_sim[inds[0], inds[1], inds[2]] = 0
    return cos_sim

def get_voxel_neighbors(matrix_shape, i, j, k):
    
    neighbors = []
    dim_i, dim_j, dim_k,_ = matrix_shape
    
    for di in range(-1, 2):   
        for dj in range(-1, 2):  
            for dk in range(-1, 2):  
                if di == 0 and dj == 0 and dk == 0:
                    continue  # Skip the central voxel
                    
                new_i, new_j, new_k = i + di, j + dj, k + dk
                
                if (0 <= new_i < dim_i) and (0 <= new_j < dim_j) and (0 <= new_k < dim_k):
                    neighbors.append((new_i, new_j, new_k))
                    
    return np.array(neighbors)