from monai.transforms import Transform
import numpy as np

class LoadNumpyd(Transform):
    # load a numpy file
    def __init__(self, keys):
        self.keys = keys
        
    def __call__(self, data):
        for key in self.keys:
            data[key] = np.load(data[key])
        return data
    
class BifNP2Dict(Transform):
    # transform the numpy bifurcation array to a dictionary with the correct keys
    def __init__(self, key):
        self.key = key # key containing the bifurcation array
        self.artis =['BA-I', 'BA-S', 'PcoA-P (R)', 'PCA-P2 (R)', 'PcoA-A (R)', 'ICA (R)', 'ACA-A1 (R)', 'MCA (R)',
                    'AcoA (R)', 'ACA-A2 (R)', 'AcoA (L)', 'ACA-A2 (L)', 'ACA-A1 (L)', 'MCA (L)', 'ICA (L)', 
                     'PcoA-A (L)', 'PcoA-P (L)', 'PCA-P2 (L)']
        
    def __call__(self, data):
        bif_dict = {}
        for i, arti in enumerate(self.artis):
            bif_dict[arti] = data[self.key][i,:]
        data[self.key] = bif_dict
        return data