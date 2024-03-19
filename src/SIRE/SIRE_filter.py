import os
import torch
import numpy as np
from tqdm import tqdm
from monai.transforms import LoadImaged, ScaleIntensityRangePercentilesd, Compose
from monai.data import ITKReader
from scipy.stats import entropy
from torch.nn.functional import softmax
from sklearn.metrics.pairwise import haversine_distances
from trimesh.creation import icosphere

from src.SIRE.utils import TrackerSphere
from src.utils.general import load_params
from src.data.transforms import LoadNumpyd, BifNP2Dict
from src.SIRE.transforms import MakeMultiscaleDataBatch, MakeMultiscaleData, CreateSIREGrid
from src.SIRE.models import GEMCNN

#import GEM transforms (to make conv in batches correct!)ö
from gem_cnn.transform.gem_precomp import GemPrecomp
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh
from torch_geometric.transforms import FaceToEdge

class ProcessSIRE3DImage:
    # a class that handles the processing of SIRE on a 3D image
    def __init__(self,
                **params):
        
        self.params = params # all the settings for processing this exp
        
        #load experiment specific params
        configfile = os.path.join(self.params['expdir'], 'train.yml')
        self.exp_params = load_params(configfile)
        
        self.params['nverts'] = icosphere(subdivisions=self.exp_params['TransformParams']['subdivisions']).vertices.shape[0]
        
        # load network and create the OnionSample transform
        self.network = self.load_network()
        self.sampler = MakeMultiscaleDataBatch(npoints=self.exp_params['TransformParams']['npoints'],
                                     n_scales=len(self.params['raylengths']),
                                     n_verts=self.params['nverts'])
        self.GEMTransform = self.get_gem_transform()
        
    def __call__(self, data, filename=None, **params):
        
        # pass in an imagefile, load + scale intensities
        self.img = self.load_img(data)
        
        self.process_points(filename=filename)
        
    def process_points(self, filename=None):        
        # process batches of points and determine directions, scales, entropy and maximum activation value
        print(f'No. iterations ', len(torch.split(self.img['grid'], 500)))
        
        #transform gridpoints to voxel coordinates!
        gridpoints = self.img['grid'] #already in voxel coordinates now!!
        
        sphere = TrackerSphere(subdivisions=self.exp_params['TransformParams']['subdivisions'])
    
        results = {'entropy': [], 'entr_norm': [], 
                   'd1': [], 'd2': [], 
                   'max_activation': [], 
                   'max_scale': [], 'avg_scale': []}
        
        for i, points in tqdm(enumerate(torch.split(gridpoints, 500))):
            if i == 0 or len(points) < 500:
                onion_batch_GEM = self.sampler(self.img, points)
                onion_batch_GEM = self.GEMTransform(onion_batch_GEM)

            else:
                onion_batch = self.sampler(self.img, points)
                onion_batch_GEM.features = onion_batch.features # copy pasta features
                
            with torch.no_grad():
                pred_batch = self.network(onion_batch_GEM.cuda()).cpu()             
                
            for j in range(points.shape[0]):
                output = self.process_pred(pred_batch[j * self.params['nverts']: (j+1) * self.params['nverts'], :], sphere)
                for key in output:
                    results[key].append(output[key])
                index_vox = np.unravel_index((500*i) + j, self.img['vectorfield'].shape[1:])
                o = torch.tensor([output['entropy'], output['max_activation'], output['max_scale']])
                self.img['vectorfield'][:, index_vox[0], index_vox[1], index_vox[2]] = torch.cat([output['d1'], output['d2'], o])
                    
        np.save(os.path.join(self.params['resultsdir'], filename + '.npy'),
                self.img['vectorfield'].numpy())
        

    def get_direction(self, prediction, sphere):
        # return directions predicted by SIRE in spherical coordinates (r, phi, theta)
        directions = []
        if torch.max(prediction) > 0:
            max_proj = torch.max(prediction, dim=1)[0]
            # first direction
            ind = torch.argmax(max_proj)
            direction_spher = sphere.sphereverts[ind, :]  # phi, theta, omit r
            dists = torch.from_numpy(haversine_distances(
                sphere.sphereverts[:, 1:] - np.array([np.pi / 2, 0]),
                direction_spher[1:].reshape(1, -1) - np.array([np.pi / 2, 0])))
            directions.append(sphere.cartverts[ind, :])

            # mask out predictions that are too far away
            max_proj[[torch.where(dists < 0.5 * np.pi)[0].long()]] = 0
            ind = torch.argmax(max_proj)
            directions.append(sphere.cartverts[ind, :])
            directions = self.order_directions(directions)
        else: # in case of indecisive SIRE output (zero everywhere), pick arbitrary direction
            index = np.random.choice(np.arange(sphere.cartverts.shape[0]))
            directions.append(sphere.cartverts[index,:])
            directions.append(-1 * sphere.cartverts[index,:])
        return directions

    @staticmethod
    def order_directions(directions):
        cos_1 = np.dot(directions[0], np.array([0,0,1]))
        cos_2 = np.dot(directions[1], np.array([0,0,1]))
        if cos_1 < cos_2 or cos_1 == cos_2:
            return [directions[1], directions[0]]
        else:
            return [directions[0], directions[1]]


    def process_pred(self, pred, sphere):
        # process a single point
        pred_max = torch.max(pred, dim=1)[0]

        entr = entropy(softmax(pred_max, dim=-1)) / np.log([len(pred_max)])[0]
        entr_new = entropy(pred_max / pred_max.sum()) / np.log([len(pred_max)])[0]
        
        if torch.max(pred)> 0:
            max_scale = self.params['raylengths'][torch.argmax(torch.max(pred, dim=0)[0])]

        else:
            max_scale = 0.01

        # get weighted average of the predictions
        activations = torch.max(pred, dim=0)[0]

        if activations.sum() > 0:
            diffs = np.diff(np.array(self.params['raylengths']))
            binsizes = np.array([diffs[0]] + (0.5 * diffs[:-1] + 0.5 * diffs[1:]).tolist() + [diffs[-1]])
            avg_scale = np.average(np.array(self.params['raylengths']), 
                                   weights=activations.numpy() * binsizes)
        else:
            avg_scale = 0.01

        directions = self.get_direction(pred, sphere)
        max_activation = torch.max(pred).item()
        return {'entropy': entr,
                'entr_norm': entr_new,
                'd1': torch.tensor(directions[0]),
                'd2': torch.tensor(directions[1]),
                'max_activation': max_activation,
                'max_scale': max_scale,
                'avg_scale': avg_scale}
        
    def load_img(self, data, *params):
        # load image and create SIRE grid already
        loader = LoadImaged(keys=['img'], 
                            image_only=False, 
                            allow_missing_keys=True)
        reader = ITKReader(reverse_indexing=True, affine_lps_to_ras=False)
        loader.register(reader)
    
        # preprocess intensities
        transform = Compose([loader,
                            LoadNumpyd(['bifs']),
                            BifNP2Dict('bifs'),
                            CreateSIREGrid('bifs',
                                           crop_margin=self.params['cropping_margin'],
                                           grid_resolution=self.params['grid_resolution']),
                            ScaleIntensityRangePercentilesd(keys=['img'],
                                                            lower=1, upper=99,
                                                            b_min=0., b_max=1.,
                                                            clip=True),
                            MakeMultiscaleData(self.params['raylengths'], 
                                         GEMGCN=False,
                                        measure=self.params['measure'],
                                        subdivisions=self.exp_params['TransformParams']['subdivisions'],
                                        npoints=self.exp_params['TransformParams']['npoints']) # gem=false because of unequal batch sizes at end of grid
                            ])
        return transform(data)
        
    def load_network(self):
        network = GEMCNN(nverts=self.params['nverts'],
                         nlayers=len(self.params['raylengths']),
                        channels=self.exp_params['NetworkParams']['channels'],
                        len_features=self.exp_params['TransformParams']['npoints'])
        state_dict = os.path.join(self.params['expdir'], 'model_weights.pt')
        network.load_state_dict(torch.load(state_dict))
        return network.cuda()
    
    @staticmethod
    def get_gem_transform():
        return Compose([
            FaceToEdge(remove_faces=False),
            compute_normals_edges_from_mesh,
            SimpleGeometry(),
            GemPrecomp(2, 2)
        ]) 