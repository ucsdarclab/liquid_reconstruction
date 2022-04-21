import torch
import torch.nn as nn



class FluidGravityForce(nn.Module):
    def __init__(self, gravity, maxSpeed = 3):
        """
        Initializes a fluid gravity model.

        Arguments:
            gravity: Gravity vector in the global frame (same as particle l) for the simulation
            maxSpeed: The maximum magnitude of the particle velocities. Higher velocities are clamped.
                      Previous work used: MAX_VEL = 0.5*0.1*NSUBSTEPS/DT
        """
        super(FluidGravityForce, self).__init__()
        self.gravity  = gravity
        self.maxSpeed = maxSpeed
        self.relu = nn.ReLU()

    def _cap_magnitude(self, A, cap):
        d = len(A.size())
        vv = torch.norm(A, 2, d-1, keepdim=True)
        vv = cap/(vv + 0.0001)
        vv = -(self.relu(-vv + 1.0) - 1.0)
        return A*vv

    def forward(self,locs, vel, dt):
        """
        Applies gravity force to fluid sim
        Inputs:
            locs: A BxNx3 tensor where B is the batch size, N is the number of particles.
                  The tensor contains the locations of every particle.
            vels: A BxNx3 tensor that contains the velocity of every particle
            dt: timestep to predict for
            gravity: 1x1x3 tensor containing the direction of gravity in the same coordinate frame as particles
            maxSpeed: maximum velocity possible for nay particle
        Returns:
            locs: A BxNx3 tensor with the new particle positions
            vel:  A BxNx3 tensor with the new particle velocities
        """
        vel = vel + self.gravity*dt
        vel = self._cap_magnitude(vel, self.maxSpeed)
        locs = locs + vel*dt
        return locs, vel

class XsphViscosity(nn.Module):
    def __init__(self, radius, viscosity=0.0001):

        """
        Initializes a fluid viscosity model.

        Arguments:
            radius: The particle interaction radius. Particles that are further than this apart do not
                    interact. Larger values for this parameter slow the simulation significantly.
            viscosity: The viscosity constant.

        """
        super(XsphViscosity, self).__init__()

        self.viscosity = viscosity
        self.radius = radius

    def forward(self, locs, vel, num_points_per_cloud = None):
        """
        Apply XSPH viscosity to point cloud. Note that the velocity of each particle is held in the "feature"
        Inputs:
            locs: BxNx3 point cloud loc of the fluid
            vel:  BxNx3 vel of point cloud of the fluid
            num_points_per_cloud: list of size B which indiciates the number of points per batch in locs.
                      This is useful for 0 paddding and having uneven number of locs.
        Returns:
            vel
        """

        # L2_matrix: BxNxNx3 gives the difference between each pair of locs per batch
        D_matrix   = locs.unsqueeze(2) - locs.unsqueeze(1)
        Vel_matrix = vel.unsqueeze(2)  - vel.unsqueeze(1)

        # L2_matrix: BxNxN gives the distance between each pair of locs per batch
        L2_matrix = torch.norm(D_matrix, p=2, dim=-1)

        # kernel: BxNxN kernel evaluated between each pair of locs per batch
        kernel_viscosity = spn.KERNEL_FN['spiky']  (L2_matrix, self.radius) * (L2_matrix < self.radius)
        kernel_density   = spn.KERNEL_FN["default"](L2_matrix, self.radius) * (L2_matrix < self.radius)

        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                kernel_density[batch, num_point:, :] = 0
                kernel_density[batch, :, num_point:] = 0

        # Density: BxN density evaluated at each loc
        density = torch.sum(kernel_density, dim=-1)
        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                density[batch, num_point:] = 0

        # velKernel: BxNxNx3 velocity multiplied by kernel divided by the density
        velKernel = Vel_matrix*(kernel_viscosity/density.unsqueeze(-1).expand_as(kernel_viscosity).transpose(1,2)
                                ).unsqueeze(-1).expand(-1, -1, -1, 3)

        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                velKernel[batch, num_point:, :] = 0
                velKernel[batch, :, num_point:] = 0
        vel = vel - self.viscosity*torch.sum(velKernel, dim=-2)

        return vel


