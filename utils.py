from mesh_to_sdf import *
import trimesh
import numpy as np

def generateSDF_approx(mesh, resolution, margin, min_bound = None, max_bound = None):

    if min_bound is None or max_bound is None:
        min_bound = np.min(np.asarray(mesh.vertices)) - margin
        max_bound = np.max(np.asarray(mesh.vertices)) + margin

    #Get the "bottom" position of the sdf array.
    position_of_sdf = np.array([min_bound, min_bound, min_bound])

    tri_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
    # sdf_tri_mesh = trimesh.proximity.ProximityQuery(tri_mesh)

    grid_pts = np.mgrid[min_bound:max_bound:resolution,
                        min_bound:max_bound:resolution,
                        min_bound:max_bound:resolution]
    target_shape = grid_pts.shape[1:]

    grid_pts = grid_pts.reshape((3,-2)).transpose() + resolution/2.0*np.ones((1,3))
    sdf_list = mesh_to_sdf(tri_mesh, grid_pts)

    sdf = -sdf_list.reshape(target_shape)

    return sdf, position_of_sdf