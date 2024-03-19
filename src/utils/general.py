import numpy as np
import yaml
import torch

def load_model_params(file):
    # load parameters from config.yaml
    np.random.seed(seed=123299)
    with open(file, 'rb') as f:
        conf = yaml.safe_load(f.read())
    f.close()
    return conf

def transform_points(points, affine):
    # transform points from one coordinate system to another through affine
    points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1).T
    return (affine @ points).T[:, :-1]

def spher2cart(spher_coords):
    # transform an Nx3 matrix of (unit length) Spherical coordinates [r, phi, theta] into an Nx3 matrix with [x,y,z] Cartesian coordinates.
    r, phi, theta = spher_coords[:, 0], spher_coords[:, 1], spher_coords[:, 2]
    if isinstance(spher_coords, np.ndarray):
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        return np.array([x, y, z]).T
    elif torch.is_tensor(spher_coords):
        cart_coords = torch.zeros_like(spher_coords)
        cart_coords[:, 0] = r * torch.cos(theta) * torch.sin(phi)
        cart_coords[:, 1] = r * torch.sin(theta) * torch.sin(phi)
        cart_coords[:, 2] = r * torch.cos(phi)
        return cart_coords


def cart2spher(cart_coords):
    # transform an Nx3 matrix of (unit length) Cartesian coordinates into an Nx3 matrix with [r, phi, theta] spherical coordinates.
    if torch.is_tensor(cart_coords):
        cart_coords /= torch.linalg.norm(cart_coords, dim=1, keepdim=True)
        coords_spherical = torch.ones_like(cart_coords)
        theta = torch.arctan2(cart_coords[:, 1], cart_coords[:, 0])
        phi = torch.arccos(cart_coords[:, 2])
        theta += np.pi * 2 * (theta < 0).long()
    else:
        cart_coords = cart_coords / np.expand_dims(
            np.linalg.norm(cart_coords, axis=1), 1
        )
        coords_spherical = np.ones_like(cart_coords)
        theta = np.arctan2(cart_coords[:, 1], cart_coords[:, 0])
        phi = np.arccos(cart_coords[:, 2])
        theta += np.pi * 2 * (theta < 0).astype(int)

    coords_spherical[:, 1] = phi
    coords_spherical[:, 2] = theta
    return coords_spherical
