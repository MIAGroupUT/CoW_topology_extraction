import copy
from monai.transforms import Transform, Compose
import numpy as np
import torch
from torch.nn.functional import grid_sample
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge
from gem_cnn.transform.gem_precomp import GemPrecomp
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh
from src.utils.general import transform_points
from src.SIRE.utils import Sphere

class MakeMultiscaleDataBatch(Transform):
    # generate a batch of multiscale spherical samples (SIRE adaptation to filter)
    
    def __init__(self, npoints=32, n_verts=642, n_scales=8):
        super().__init__()
        self.npoints = npoints
        self.n_verts = n_verts
        self.n_scales = n_scales

    def __call__(self, data, centers):
        n_centers = centers.shape[0]
        
        # stack rays to create batch
        rays = torch.tile(data['onion'].coords, [n_centers, 1])
        centers_stacked = torch.repeat_interleave(centers, data['onion'].coords.shape[0], dim=0)
        
        rays = rays + centers_stacked
        # normalize rays to [-1, 1] domain
        rays = (rays * (2 / torch.flip(torch.tensor(data['img'].shape), dims=[0])) - 1).float()

        onion = copy.deepcopy(data['onion'])

        onion.features = grid_sample(data['img'].unsqueeze(0).unsqueeze(0), 
                rays.view(1, 1, 1, rays.shape[0], 3), padding_mode='reflection', align_corners=True).squeeze().reshape(-1, self.npoints)
        
        # fix pos and faces for batching!!
        onion.pos = torch.tile(onion.pos, [n_centers, 1])
        addto = torch.repeat_interleave(torch.arange(n_centers) * self.n_verts * self.n_scales, onion.face.shape[1], dim=0)
        onion.face = torch.tile(onion.face, [1, n_centers]) 
        onion.face += addto
        return onion

class MakeMultiscaleData(Transform):
    """
    Dictionary-based transform that creates the multiscale spherical samples given spheres in voxel coordinates of the respective scan
    Make the stacked faces / pos vector for the PyG data
    """

    def __init__(
        self,
        raylenghts: list,
        npoints=32,
        measure="cm",
        subdivisions=3,
        GEMGCN=False,
        randomize_scales=False,
    ):
        """
        :param raylengths: radii of the considered scales
        :param npoints: number of points along each ray
        :param measure: unit of measurement in spacing of the image
        :param subdivisions: number of subdivisions in the ICOsphere

        """
        super().__init__()
        self.raylengths = raylenghts
        self.measure = measure
        self.sphere = Sphere(subdivisions=subdivisions)
        self.npoints = npoints
        self.FaceToEdge = FaceToEdge(remove_faces=False)
        self.GEM = GEMGCN
        if self.GEM: #COMMENT FOR NOW
            self.GEMtransform = Compose([compute_normals_edges_from_mesh,
                                         SimpleGeometry(),
                                         GemPrecomp(2, 2)
                                         ])
        self.randomize_scales = randomize_scales

    def __call__(self, data):
        # get the onion ray coordinates in voxelcoordinates! (tensor)
        onion = []
        for length in self.raylengths:
            if self.measure == "cm":
                onion.append(
                    torch.from_numpy(
                        self.sphere.get_rays(self.npoints, length / 10)
                    ).float()
                )
            else:  # affine is in mm, so do NOT divide raylength by 10!!!
                onion.append(
                    torch.from_numpy(self.sphere.get_rays(self.npoints, length)).float()
                )
        onion = torch.cat(onion, dim=0)
        origin = data["img_meta_dict"]["affine"][:-1, -1]
        onion += origin
        onion = transform_points(
            onion, torch.linalg.inv(data["img_meta_dict"]["affine"])
        )
        pos, facestack = self._structure_faces()
        data["onion"] = Data(coords=onion, face=facestack, pos=pos)
        if self.randomize_scales:
            data["onion"].OG_coords = copy.deepcopy(onion)

        data["onion"] = self.FaceToEdge(data["onion"])
        if self.GEM:  # perform additional transforms
            data['onion'] = self.GEMtransform(data['onion'])
        return data

    def _structure_faces(self):
        # stack faces and positions to structure the graph
        faces = torch.from_numpy(
            self.sphere.sphere.faces.T
        ).long()
        pos = torch.from_numpy(self.sphere.sphere.vertices).float()
        facestack = torch.hstack(
            [faces + i * pos.shape[0] for i in range(len(self.raylengths))]
        )
        pos = torch.vstack([pos] * len(self.raylengths))
        return pos, facestack
    

class CreateSIREGrid(Transform):
    """
    Create the grid to evaluate SIRE on, based on a fixed resolution (in mms)
    crop based on predicted bifurcation locations
    Find location of bifurcations of interest (BoIs) in this grid 
    """
    def __init__(self, bifurcation_key, grid_key='grid', crop_margin=2., grid_resolution=0.5):
        """
        bifurcation_key: key that contains the bifurcation dictionary
        crop_margin: margin for cropping [mm]
        grid_resolution: resolution of the SIRE grid [mm]
        """
        self.bifurcation_key = bifurcation_key
        self.crop_margin = crop_margin
        self.grid_resolution = grid_resolution
        # bifurcations of interest
        self.BoIs = ['AcoA (L)', 'AcoA (R)', 'ACA-A1 (R)', 'ACA-A1 (L)',
                     'PcoA-A (L)', 'PcoA-A (R)', 'PcoA-P (L)', 
                     'PcoA-P (R)', 'BA-S'] #'BA-I'
        self.grid_key = grid_key
    
    def __call__(self, data):
        BoI_locs = np.zeros([len(self.BoIs), 3])
        for i, boi in enumerate(self.BoIs):
            # locations in world coordinates
            BoI_locs[i,:]=data[self.bifurcation_key][boi]
        data['BoIs'] = {'world': torch.from_numpy(BoI_locs).float()} # bifurcation locations in (world coordinates)
        data['BoIs']['voxel'] = transform_points(data['BoIs']['world'],
                                                torch.linalg.inv(data['img_meta_dict']['affine']))
        stepsize = np.array([self.grid_resolution] * 3) / data['img_meta_dict']['spacing']
        crop_margin = np.array([self.crop_margin] * 3) / data['img_meta_dict']['spacing']
        
        # find bounding boxes (VOXEL coordinates)
        min_point = np.min(data['BoIs']['voxel'].numpy(), axis=0) - crop_margin
        max_point = np.max(data['BoIs']['voxel'].numpy(), axis=0) + crop_margin
        data[f'{self.grid_key}_meta'] = {'origin': transform_points(np.reshape(min_point, [1,3]),
                                                        data['img_meta_dict']['affine']).squeeze().numpy(),
                            'spacing': [self.grid_resolution] * 3}
        # create affine matrix of the subsampled grid
        affine = data['img_meta_dict']['affine'][:3, :3].numpy()
        D = affine @ np.linalg.inv(np.eye(3) * data['img_meta_dict']['spacing'])
        new_spacing = np.array([self.grid_resolution] * 3) * np.eye(3)
        new_affine = np.zeros([4,4])
        new_affine[:3,:3] = D @ new_spacing
        new_affine[:-1, -1] = data[f'{self.grid_key}_meta']['origin']
        new_affine[-1,-1] = 1
        data[f'{self.grid_key}_meta']['affine'] = new_affine
        
        # build grid to evaluate SIRE on
        xx = torch.arange(min_point[0], max_point[0], stepsize[0])
        yy = torch.arange(min_point[1], max_point[1], stepsize[1])
        zz = torch.arange(min_point[2], max_point[2], stepsize[2])
        X, Y, Z = torch.meshgrid([xx,yy,zz], indexing='ij')
        # VOXEL coordinates
        data[self.grid_key] = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)

        data['vectorfield'] = torch.zeros([9, X.shape[0], X.shape[1], X.shape[2]])
        data[f'{self.grid_key}_meta']['shape'] = data['vectorfield'].shape[1:]
        
        data['BoIs'][self.grid_key] = torch.zeros_like(data['BoIs']['world'])
        for i, BoI in enumerate(data['BoIs']['voxel']):
            x_coord = torch.argmin(torch.abs(xx - BoI[0]))
            y_coord = torch.argmin(torch.abs(yy - BoI[1]))
            z_coord = torch.argmin(torch.abs(zz - BoI[2]))
            data['BoIs'][self.grid_key][i,:] = torch.tensor([x_coord, y_coord, z_coord])
        return data