import glob
import os
from src.SIRE.SIRE_filter import ProcessSIRE3DImage
from src.dijkstra.shortest_path import ExtractPaths
from pdb import set_trace as bp

def main(data, **params):
    SIREFilter = ProcessSIRE3DImage(**params['SIRE_filter'])
    PathExtractor = ExtractPaths(**params['Dijkstra'])
    if params['Dijkstra']['vis']:
        data['img_OG'] = data['img']

    # vectorfield with SIRE output is under key 'vectorfield' in data
    data = SIREFilter(data)
    bp()
    paths = PathExtractor(data)
    # RF classifier here!

if __name__ == "__main__":
    datadir = '../data'
    pt_ID = 'E_30-004' #patient ID as given in data folder

    params = {'SIRE_filter': {
                'expdir': '../data/sire-weights',
                'raylengths': [1, 2, 5, 7], # scales for SIRE (in mm)
                'grid_resolution': 0.5, # resolution of SIRE grid
                'cropping_margin': 5, # margin around the predicted CoW bifurcations
                'measure': 'mm',
                'gridmode': 'uniform'
                },
            'Dijkstra': {
                'c_img': 0, 'c_entr': 0, 'c_orient': 1,
                'vis': True, 
                'grid_resolution': 0.5
                }
            }

    # make dictionary with paths to scan and predicted bifurcation locations
    imgfiles = glob.glob(os.path.join(datadir, 'scans', f'{pt_ID}*.nii.gz'))
    bif_files = glob.glob(os.path.join(datadir, 'bifurcations', f'{pt_ID}*.npy'))
    if len(imgfiles) > 0 and len(bif_files) > 0:
        data = {'bifs': bif_files[0],
                'img': imgfiles[0]}
        main(data, **params)
    else:
        print(f'No image or bifurcation files found in {datadir}')