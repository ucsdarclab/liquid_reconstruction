'''
Example code of simulating a liquid fountain in a simple box.
The simulation is based on PBF (https://mmacklin.com/pbf_sig_preprint.pdf)
Note: the same functions used for this PBF are used for liquid reconstruction
'''

import torch
import open3d as o3d
from differentiableFluidSim import FluidGravityForce, uniformSource,\
                                   XsphViscosity, MullerConstraints
from utils import generateSDF_approx

# Simulation parameters:
dt = 1/60.0
num_particles_emitted_per_frame = 1
gravity_force = [0, 0, 9.8]
max_particle_velocity = 2
interaction_radius = 0.005
sdf_resolution = 0.05
dampening_factor = 0.01


# Callback for open3d visualizer to end simulation
end_simulation = False
def close_visualizer(vis):
    global end_simulation
    end_simulation = True


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Generate box mesh
    mesh = o3d.geometry.TriangleMesh.create_box()
    wire_box = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

    # Get SDF from mesh
    print("Generating SDF mesh. This can take a while...")
    sdf, pos_sdf = generateSDF_approx(mesh, sdf_resolution, 50*sdf_resolution)
    
    # Prepare simulation modules
    fluid_gravity_force = FluidGravityForce(torch.tensor(gravity_force, 
                                                        dtype=torch.float32).reshape((1, 1, -1)).cuda(),
                                            maxSpeed=max_particle_velocity).cuda()

    collision_pose = torch.tensor([[[pos_sdf[0], pos_sdf[1], pos_sdf[2], 
                                    0, 0, 0, 1]]], dtype=torch.float32).cuda()
    fluid_constraints = MullerConstraints(torch.from_numpy(sdf), sdf_resolution, 
                                          collision_pose, radius=0.005,
                                          numStaticIterations=3, numIteration=5,
                                          fluidRestDistance = 0.6).cuda()

    fluid_viscosity = XsphViscosity(radius=interaction_radius).cuda()

    # Source location of liquid:
    source_init_loc_torch = torch.tensor([0.5, 0.05, 0.8], dtype=torch.float32).reshape((1, 1, -1)).cuda()
    source_init_vel_torch = torch.tensor([0.0,0.0,0.0], dtype=torch.float32).reshape((1, 1, -1)).cuda()
    locs, vel = uniformSource(source_init_loc_torch, source_init_vel_torch,
                              num_particles_emitted_per_frame, 1, 0.01)

    # Visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_callback(88, close_visualizer)

    # Add all the geometry to visualizer
    wire_box = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    vis.add_geometry(wire_box)
    pcd_fluid = o3d.geometry.PointCloud()
    pcd_fluid.points = o3d.utility.Vector3dVector(locs.cpu().numpy()[0])
    vis.add_geometry(pcd_fluid)

    print("Hit X to close simulation")
    while not end_simulation:

        # Predict particles according to their velocity and gravity
        new_locs, vel = fluid_gravity_force(locs, vel, dt)

        # Add new particles from source
        add_locs, add_vel = uniformSource(source_init_loc_torch, 
                                          source_init_vel_torch,
                                          num_particles_emitted_per_frame, 1, 0.01)

        new_locs = torch.cat((new_locs, add_locs), 1)
        vel      = torch.cat((vel,  add_vel), 1)


        # Apply fluid position constriants
        new_locs = fluid_constraints(locs = new_locs)

        # Update velocity after applying position constraints to the old particles
        vel[:, :locs.shape[1], :] = (1-dampening_factor)*(new_locs[:, :locs.shape[1], :] - locs)/dt

        # Apply fluid viscosity
        vel = fluid_viscosity(new_locs, vel)

        locs = new_locs

        # Update visualizer
        pcd_fluid.points  = o3d.utility.Vector3dVector(locs.cpu().numpy()[0])
        pcd_fluid.paint_uniform_color([0.5, 0.5, 1.0])
        vis.update_geometry(pcd_fluid)

        vis.poll_events()
        vis.update_renderer()


