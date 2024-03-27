from src.dijkstra.transforms import GetStartEndPoints, InterpolateImage, get_cos_similarity
from src.SIRE.transforms import CreateSIREGrid
from src.utils.general import transform_points
from dijkstra3d import dijkstra
from monai.transforms import Compose, LoadImaged
from monai.data import ITKReader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from pdb import set_trace as bp

class ExtractPaths():
    # extract the artery paths from the cost functions
    def __init__(self, **params):
        self.params = params
        self.transform = self.get_transform()        

    def get_transform(self):
        transform = []
        if self.params['vis']:
            loader = LoadImaged(['img_OG'], image_only=False)
            loader.register(ITKReader(reverse_indexing=True, affine_lps_to_ras=False))
            transform += [loader,
                          CreateSIREGrid('bifs',
                                         crop_margin=5,
                                         grid_resolution=0.3, 
                                         grid_key='MIP_grid')]
        transform +=[
                CreateSIREGrid('bifs', crop_margin=5, grid_resolution=self.params['grid_resolution']),
                GetStartEndPoints('BoIs'),
                InterpolateImage('img', 'grid')]
        return Compose(transform)

    def get_cost_function(self, data):
        print('constructing cost functions...')
        c_img = np.clip(1 - data['int_img'], 0, 1)
        c_entr = data['vectorfield'][6,:,:,:].numpy()
        cossim = get_cos_similarity(data['vectorfield'])
        c_orient = 1 - ((cossim + 1) / 2)
        bp()
        return self.params['c_img'] * c_img + self.params['c_entr'] * c_entr + self.params['c_orient'] * c_orient

    def make_MIP_plot(self, data, paths):
        colors = plt.get_cmap(name='tab10')
        fig = plt.figure()
        T = InterpolateImage('img_OG', 'MIP_grid')
        sample = T(data)
        MIP = np.max(data['int_img_OG'], axis=2)
        plt.imshow(MIP.T, 'gray')
        plt.axis('off')
        # plot centerlines
        for j, arti in enumerate(list(paths)):
            # transform to different grid coordinates
            p_vox = transform_points(paths[arti], np.linalg.inv(sample['MIP_grid_meta']['affine']))
            plt.plot(p_vox[:,0], p_vox[:,1], color = colors(j),
                    path_effects=[pe.Stroke(linewidth=4, foreground='w'), pe.Normal()])
        # plot bifurcation points
        bifs_MIP = transform_points(sample['BoIs']['world'], np.linalg.inv(sample['MIP_grid_meta']['affine']))
        plt.plot(bifs_MIP[:,0], bifs_MIP[:,1], 
                '.', markersize=15,markerfacecolor='w', markeredgecolor='k', alpha=0.7)
        plt.show()

    def __call__(self, data):
        data = self.transform(data)
        cost_function = self.get_cost_function(data)
        clines = {}
        # loop over arteries
        for arti in list(data['seedpoints']):
            print(f'now tracking {arti}')
            start, end = data['seedpoints'][arti]
            # on grid points from SIRE grid!
            path = dijkstra(cost_function, start, end, connectivity=26)
            # transform to world coordinates
            clines[arti] = transform_points(path, data['grid_meta']['affine'])
        if self.params['vis']:
            self.make_MIP_plot(data, clines)
        return clines

