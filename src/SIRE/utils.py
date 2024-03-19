import numpy as np
from trimesh import icosphere
from src.utils.general import cart2spher

class Sphere:
    # this is the sphere object, that is used for obtaining the right data for the GEM-CNN.
    # it contains the points where the image intensities are sampled for each vertex in the mesh

    def __init__(self, subdivisions=3):  # nverts = 642
        self.sphere = icosphere(subdivisions=subdivisions)
        self.sphereverts = cart2spher(self.sphere.vertices)
        self.cartverts = self.sphere.vertices

    def get_rays(self, npoints, ray_length, center=np.array([[0, 0, 0]])):
        """
        transform the Nx3 matrix containing the spherical coordinates of the vertices into image coordinates
        of all the points on the rays.

        Args:
            npoints: number of points on the ray
            ray_length:  real-world length of ray (in mms/cms)
            center: center of sphere, in world-coordinates

        Returns: (Nxraylength) x 3 matrix containing cartesian coordinates

        """
        rays = np.linspace(0, ray_length, npoints)
        sphereverts_long = self.sphereverts.repeat(npoints, axis=0)
        cart_coords = np.ones([self.sphereverts.shape[0] * npoints, 3])
        cart_coords[:, 0] = (
            np.tile(rays, self.sphereverts.shape[0])
            * np.sin(sphereverts_long[:, 1])
            * np.cos(sphereverts_long[:, 2])
        )
        cart_coords[:, 1] = (
            np.tile(rays, self.sphereverts.shape[0])
            * np.sin(sphereverts_long[:, 1])
            * np.sin(sphereverts_long[:, 2])
        )
        cart_coords[:, 2] = np.tile(rays, self.sphereverts.shape[0]) * np.cos(
            sphereverts_long[:, 1]
        )
        return cart_coords + center  # [x, y, z] in world-coordinates
