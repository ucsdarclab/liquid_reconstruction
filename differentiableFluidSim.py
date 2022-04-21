import torch
import torch.nn as nn
import numpy as np

import SmoothParticleNets as spn
import sympy


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
    def __init__(self, radius, viscosity=0.01):

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

class GradConvSDF(nn.Module):
    def __init__(self, sdfs, sdf_sizes, ndim, max_distance, dilation= 1.0 / 100.0, kernel_size = 3):
        super(GradConvSDF, self).__init__()
        self.ndim = ndim
        self.convsdfgrad = []
        for d in range(ndim):
            ks = [1] * ndim
            ks[d] = kernel_size
            convsdf = spn.ConvSDF(sdfs, sdf_sizes, 1, ndim, kernel_size=ks, dilation=dilation,
                                  max_distance=max_distance, with_params=False, compute_pose_grads=False)
            convsdf.weight.data.fill_(0)

            # Create finite difference weights
            weights = sympy.calculus.finite_diff.finite_diff_weights(1, np.arange(-int(kernel_size/2),
                                                                                    int(kernel_size/2)+1) * dilation,
                                                                      0)[1][-1]
            # Done in a loop because of weird data type from sympy
            # also can't use enumerate for some reason... I should learn about sympy
            for idx in range(kernel_size):
                convsdf.weight.data[0, idx ] = float(weights[idx])

            convsdf.bias.data.fill_(0)
            self.convsdfgrad.append(convsdf)
            exec("self.convsdfgrad%d = convsdf" % d)

    def forward(self, locs, idxs, poses, scales):
        return torch.cat([self.convsdfgrad[d](locs, idxs, poses, scales)
                          for d in range(self.ndim)], 2)

class FluidConstraints(nn.Module):
    def __init__(self, sdf, sdf_resolution, sdf_pose, radius, fluidRestDistance = 0.6, pressure_kernel="default",
                 derivative_of_pressure_kernel="dspiky", dilation_for_conv_sdf = 1/100.0, kernel_size_for_sdf = 3):
        """
        Initializes a fluid position constraint model.

        Arguments:
            sdf:  SDF of mesh in environment. This argument is passed directly as the sdfs
                  argument of ConvSDF. Refer to the documentation of that layer for details.
            sdf_resolution: The size (aka resolution) for the sdfs argument. This argument
                            is also passed directly to ConvSDF. Refer to the documentation for that layer
                            for details.
            sdf_pose: Pose of the sdf (Previously inputted to forward in fluid_sim.py)
            radius: The particle interaction radius. Particles that are further than this apart do not
                    interact. Larger values for this parameter slow the simulation significantly.
            fluidRestDistance: The distance fluid particles should be from each other at rest, expressed
                               as a ratio of the radius.
           pressure_kernel: Kernel from spnet used to evaluate density/pressure. For list of kernels check out:
                https://github.com/cschenck/SmoothParticleNets/blob/master/python/SmoothParticleNets/kernels.py
            dilation_for_conv_sdf: dilation/step size for the gradient of sdf used in ConvSDF
            kernel_size_for_sdf: Size of kernel for sdf

        """


        super(FluidConstraints, self).__init__()

        self.stiffness = 0.0
        self.density_rest = 0.0
        self.fluidRestDistance = fluidRestDistance*radius
        self.radius = radius
        self.pressure_kernel = pressure_kernel
        self.derivative_of_pressure_kernel = derivative_of_pressure_kernel
        self._calculate_rest_density(self.fluidRestDistance)


        #All the differential distance to collision stuff
        self.idxs = torch.zeros(1, 1).cuda()
        self.scales = torch.ones(1, 1).cuda()
        self.sdf_pose = sdf_pose

        max_distance = torch.abs(sdf).max().item()
        self.convsdfgrad = GradConvSDF([sdf], [sdf_resolution], 3,
                                       max_distance=max_distance, dilation=dilation_for_conv_sdf,
                                       kernel_size=kernel_size_for_sdf)
        self.convsdfcol = spn.ConvSDF([sdf], [sdf_resolution], 1, 3, 1, 1,
                                      max_distance=max_distance, with_params=False, compute_pose_grads=False)
        self.convsdfcol.weight.data.fill_(-1)
        self.convsdfcol.bias.data.fill_(0)
        self.relu = nn.ReLU()

    #Given the minimum distance between particles (fluidRestDistance) (similar to density constraint)
    #Calculate the resting density of the fluid
    #Also compute derivative of this?? Corresponds to stiffness...
    def _calculate_rest_density(self, fluidRestDistance):
        points = np.array(self._tight_pack3D(self.radius, fluidRestDistance, 2048))
        d = np.sqrt(np.sum(points**2, axis=1))
        rho = 0
        rhoderiv = 0
        for dd in d:
            rho += spn.KERNEL_FN[self.pressure_kernel](dd, self.radius)
            rhoderiv += spn.KERNEL_FN[self.derivative_of_pressure_kernel](dd, self.radius)**2
        self.density_rest = float(rho)
        self.stiffness = float(1.0/rhoderiv)

    # Generates an optimally dense sphere packing at the origin (implicit sphere at the origin)
    def _tight_pack3D(self, radius, separation, maxPoints):
        dim = int(np.ceil(1.0*radius/separation))
        points = []
        for z in range(-dim, dim+1):
            for y in range(-dim, dim+1):
                for x in range(-dim, dim+1):
                    xpos = x*separation + \
                        (separation*0.5 if ((y+z) & 1) else 0.0)
                    ypos = y*np.sqrt(0.75)*separation
                    zpos = z*np.sqrt(0.75)*separation

                    # skip center
                    if xpos**2 + ypos**2 + zpos**2 == 0.0:
                        continue

                    if len(points) < maxPoints and np.sqrt(xpos**2 + ypos**2 + zpos**2) <= radius:
                        points.append([xpos, ypos, zpos])
        return points

    def _fix_static_collisions(self, locs, idxs, poses, scales, collisionDistance):

        if locs.dtype != torch.float32:
            raise Warning("Type for locs in _fix_static_collisions should be torch.float32 is: {}".format(locs.type))

        ret = locs

        mtd = self.convsdfcol(ret, idxs, poses, scales) + collisionDistance
        intersect = self.relu(
            mtd)#  + self.relu(-self.relu(-(mtd - 0.5)) + 0.5)*0.0
        sdfgrad = self.convsdfgrad(ret, idxs, poses, scales)
        sdfgrad = torch.nn.functional.normalize(sdfgrad, dim=-1, eps=1e-5)
        ret = ret + intersect*sdfgrad
        return ret

    def computeDensity(self, locs, num_points_per_cloud = None):
        D_matrix  = locs.unsqueeze(2) - locs.unsqueeze(1)

        # L2_matrix: BxNxN gives the distance between each pair of locs per batch
        L2_matrix = torch.norm(D_matrix, p=2, dim=-1)

        # kernel: BxNxN kernel evaluated between each pair of locs per batch
        kernel = spn.KERNEL_FN[self.pressure_kernel](L2_matrix, self.radius) * (L2_matrix < self.radius)
        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                kernel[batch, num_point:, :] = 0
                kernel[batch, :, num_point:] = 0

        # C_vector: BxN is the 0 constraint = rho_i/rho_0 - 1
        out_vector = torch.sum(kernel, dim=-1)
        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                out_vector[batch, num_point:] = 0

        return out_vector

    # i-th output computes l1 sum of density constraint at every loc after removing loc[i]
    def evaluateDensityConstraintAfterRemovingParticle(self, locs, num_points_per_cloud = None):
        D_matrix  = locs.unsqueeze(2) - locs.unsqueeze(1)

        # L2_matrix: BxNxN gives the distance between each pair of locs per batch
        L2_matrix = torch.norm(D_matrix, p=2, dim=-1)

        # kernel: BxNxN kernel evaluated between each pair of locs per batch
        kernel = spn.KERNEL_FN[self.pressure_kernel](L2_matrix, self.radius) * (L2_matrix < self.radius)
        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                kernel[batch, num_point:, :] = 0
                kernel[batch, :, num_point:] = 0

        # C_vector: BxN1 is the pre-particle removal constraint = rho_k/rho_0 - 1
        # where B : batches,
        #       N1 : k = 1, ..., N
        C_vector = torch.sum(kernel, dim=-1) / self.density_rest - 1
        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                C_vector[batch, num_point:] = 0

        # C_vector_new: BxN1xN2 is constraint after removing particle = rho_k/rho_0 - 1 - W(||p_k - p_i||)/rho_0
        # where B : batches,
        #       N1 : i = 1, ..., N
        #       N2 : k = 1, ..., N
        C_vector_new = C_vector.unsqueeze(-2).expand_as(kernel) - kernel / self.density_rest

        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                C_vector_new[batch, num_point:] = 0

        return torch.sum(torch.abs(C_vector_new), dim=-1)

    # i-th output computes l1 sum of density constraint at every loc after duplicating loc[i]
    def evaluateDensityConstraintAfterDuplicatingParticle(self, locs, num_points_per_cloud = None):
        D_matrix  = locs.unsqueeze(2) - locs.unsqueeze(1)

        # L2_matrix: BxNxN gives the distance between each pair of locs per batch
        L2_matrix = torch.norm(D_matrix, p=2, dim=-1)

        # kernel: BxNxN kernel evaluated between each pair of locs per batch
        kernel = spn.KERNEL_FN[self.pressure_kernel](L2_matrix, self.radius) * (L2_matrix < self.radius)
        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                kernel[batch, num_point:, :] = 0
                kernel[batch, :, num_point:] = 0

        # C_vector: BxN1 is the pre-particle removal constraint = rho_k/rho_0 - 1
        # where B : batches,
        #       N1 : k = 1, ..., N
        C_vector = torch.sum(kernel, dim=-1) / self.density_rest - 1
        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                C_vector[batch, num_point:] = 0

        # C_vector_new: BxN1xN2 is constraint after removing particle = rho_k/rho_0 - 1 + W(||p_k - p_i||)/rho_0
        # where B : batches,
        #       N1 : i = 1, ..., N
        #       N2 : k = 1, ..., N
        C_vector_new = C_vector.unsqueeze(-2).expand_as(kernel) + kernel / self.density_rest

        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                C_vector_new[batch, num_point:] = 0

        ind = np.diag_indices(kernel.shape[1])
        return torch.sum(torch.abs(C_vector_new), dim=-1) + torch.abs(kernel[:, ind[0], ind[1]] / self.density_rest)

    def computeMeanDensityConstraint(self, locs, num_points_per_cloud = None):
        D_matrix  = locs.unsqueeze(2) - locs.unsqueeze(1)

        # L2_matrix: BxNxN gives the distance between each pair of locs per batch
        L2_matrix = torch.norm(D_matrix, p=2, dim=-1)

        # kernel: BxNxN kernel evaluated between each pair of locs per batch
        kernel = spn.KERNEL_FN[self.pressure_kernel](L2_matrix, self.radius) * (L2_matrix < self.radius)
        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                kernel[batch, num_point:, :] = 0
                kernel[batch, :, num_point:] = 0

        # C_vector: BxN is the 0 constraint = rho_i/rho_0 - 1
        C_vector = torch.sum(kernel, dim=-1) / self.density_rest - 1
        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                C_vector[batch, num_point:] = 0
            mean_density = torch.sum(C_vector, dim=-1)/num_points_per_cloud
        else:
            mean_density = torch.mean(C_vector, dim=-1)

        return mean_density

    def computeDensityConstraint(self, locs, num_points_per_cloud = None):
        D_matrix  = locs.unsqueeze(2) - locs.unsqueeze(1)

        # L2_matrix: BxNxN gives the distance between each pair of locs per batch
        L2_matrix = torch.norm(D_matrix, p=2, dim=-1)

        # kernel: BxNxN kernel evaluated between each pair of locs per batch
        kernel = spn.KERNEL_FN[self.pressure_kernel](L2_matrix, self.radius) * (L2_matrix < self.radius)
        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                kernel[batch, num_point:, :] = 0
                kernel[batch, :, num_point:] = 0

        # C_vector: BxN is the 0 constraint = rho_i/rho_0 - 1
        C_vector = torch.sum(kernel, dim=-1) / self.density_rest - 1
        if num_points_per_cloud is not None:
            for batch, num_point in enumerate(num_points_per_cloud):
                C_vector[batch, num_point:] = 0

        return C_vector

    def forward(self, locs, num_points_per_cloud = None):
        raise NotImplementedError

    def load(self, state_dict):
        self.load_state_dict(state_dict)

class MullerConstraints(FluidConstraints):
    def __init__(self, sdf, sdf_resolution, sdf_pose, radius, numIteration=3,
                 relaxationFactor=100.0, collisionDistance=0.00125, numStaticIterations=1,
                 fluidRestDistance = 0.6, s_corr_k = 0.1, s_corr_n = 4, s_corr_q=0.1*0.2,
                 pressure_kernel="default", derivative_of_pressure_kernel="dspiky",
                 dilation_for_conv_sdf = 1/100.0, kernel_size_for_sdf = 3,
                 apply_density_constraint = True, apply_collision_constraint = True
                 ):
        """
        Initializes a fluid position constraint model.

        Arguments:
            sdf:  SDF of mesh in environment. This argument is passed directly as the sdfs
                  argument of ConvSDF. Refer to the documentation of that layer for details.
            sdf_resolution: The size (aka resolution) for the sdfs argument. This argument
                            is also passed directy to ConvSDF. Refer to the documentation for that layer
                            for details.
            sdf_pose: Pose of the sdf
            radius: The particle interaction radius. Particles that are further than this apart do not
                    interact. Larger values for this parameter slow the simulation significantly.
            numIteration: The number of constraint solver iterations to do per simulation step.
            relaxationFactor: epsilon in eq. 10 and 11 from https://mmacklin.com/pbf_sig_preprint.pdf
            collisionDistance: When a particle is closer than this to an object, it is considered colliding.
            numStaticIterations: When moving particles or objects, this is the number of substeps that
                                 movement is split into to check for collisions. The higher this value, the
                                 less likely it is that particles will clip through objects but the slower
                                 the simulation is.
            fluidRestDistance: The distance fluid particles should be from each other at rest, expressed
                               as a ratio of the radius.
            s_corr_k: All s_corr constants are for a an artifical pressure term. See:
                        https://mmacklin.com/pbf_sig_preprint.pdf section 4 for information.
                        This variable is float k in equation 13
            s_corr_n: This variable is integer n in equation 13
            s_corr_q: This variable is float delta_q / radius from equation 13 (so normalized by interaction radius)
            pressure_kernel: Kernel from spnet used to evaluate density/pressure. For list of kernels check out:
                            https://github.com/cschenck/SmoothParticleNets/blob/master/python/SmoothParticleNets/kernels.py
            derivative_of_pressure_kernel: Kernel from spnet used to evaluate derivative of pressure/density for
                                           optimization (i.e. ensuring constant pressure/density)
            dilation_for_conv_sdf: dilation/step size for the gradient of sdf used in ConvSDF
            kernel_size_for_sdf: Size of kernel for sdf
        """

        super(MullerConstraints, self).__init__(sdf, sdf_resolution, sdf_pose, radius, fluidRestDistance,
                                                pressure_kernel, derivative_of_pressure_kernel,
                                                dilation_for_conv_sdf, kernel_size_for_sdf)
        # Save all the parameters
        self.numIteration = numIteration
        self.relaxationFactor= relaxationFactor
        self.collisionDistance = collisionDistance
        self.numIterations = numIteration
        self.numStaticIterations = numStaticIterations

        self.derivative_of_pressure_kernel = derivative_of_pressure_kernel

        self.s_corr_k = s_corr_k
        self.s_corr_n = s_corr_n
        self.s_corr_q_kernel = spn.KERNEL_FN[self.pressure_kernel](s_corr_q*self.radius, self.radius)

        self._apply_density   = apply_density_constraint
        self._apply_collision = apply_collision_constraint

    def forward(self, locs, num_points_per_cloud = None):
        """
        Apply fluid constraints to locs.
        It takes as input the current fluid position state (locs).
        The return is adjusted locs to satisfy fluid constriants e.g. density and collision

        Inputs:
            locs: A BxNxD tensor where B is the batch size, N is the number of particles, and
                  D is the dimensionality of the coordinate space. The tensor contains the
                  locations of every particle.
            num_points_per_cloud: list of size B which indiciates the number of points per batch in locs.
                                  This is useful for 0 paddding and having uneven number of locs.

        Returns:
            locs: A BxNxD tensor with the updated positions of all the particles.
        """

        # Kernel decisions (poly6 and spiky) were chosen from PBF paper
        for iteration in range(self.numIterations):
            if self._apply_density:
                # D_matrix: BxNxNx3 gives the difference between each pair of locs per batch
                D_matrix  = locs.unsqueeze(2) - locs.unsqueeze(1)

                # L2_matrix: BxNxN gives the distance between each pair of locs per batch
                L2_matrix = torch.norm(D_matrix, p=2, dim=-1)

                # kernel: BxNxN kernel evaluated between each pair of locs per batch
                kernel = spn.KERNEL_FN[self.pressure_kernel](L2_matrix, self.radius) * (L2_matrix < self.radius)
                if num_points_per_cloud is not None:
                    for batch, num_point in enumerate(num_points_per_cloud):
                        kernel[batch, num_point:, :] = 0
                        kernel[batch, :, num_point:] = 0

                # C_vector: BxN is the 0 constraint = rho_i/rho_0 - 1
                C_vector = torch.sum(kernel, dim=-1) / self.density_rest - 1
                if num_points_per_cloud is not None:
                    for batch, num_point in enumerate(num_points_per_cloud):
                        C_vector[batch, num_point:] = 0
                # dKernel: BxNxN dKernel kernel normalized evaluated between each pair of locs per batch
                dKernel   = spn.KERNEL_FN[self.derivative_of_pressure_kernel](L2_matrix, self.radius) / L2_matrix
                ind = np.diag_indices(L2_matrix.shape[1])
                dKernel[:, ind[0], ind[1]] = 0  #Derivative of kernel at p_i - p_i should always be 0?
                dKernel = dKernel * (L2_matrix < self.radius)
                # dKernel: BxNxNx3 repeated dKernel on last dimension 3 times times D_matrix
                #   dSpiky_i,j = (p_i - p_j) *dspiky(d_i,j) / d_i,j
                #   Should be equivalent to grad W (p_i - p_j) from PBF paper ??
                dSpiky_3_times_D = dKernel.unsqueeze(-1).expand(-1, -1, -1, 3) * D_matrix
                if num_points_per_cloud is not None:
                    for batch, num_point in enumerate(num_points_per_cloud):
                        dSpiky_3_times_D[batch, num_point:, :, :] = 0
                        dSpiky_3_times_D[batch, :, num_point:, :] = 0

                # dC_dLocs: BxNxNx3 derivative of the constraint w.r.t. locs
                # compute non-diagonal elements
                dC_dLocs = -dSpiky_3_times_D
                # compute diagonal elements (dCi_dLocsi: BxNx3)
                dCi_dLocsi = torch.sum(dSpiky_3_times_D, dim=-2)
                dC_dLocs[:, ind[0], ind[1], :] = dCi_dLocsi
                dC_dLocs = dC_dLocs/self.density_rest


                # Reshape so we can do our newton optimization
                dC_dLocs_xyz = dC_dLocs.reshape((dC_dLocs.shape[0], dC_dLocs.shape[1], dC_dLocs.shape[2]*dC_dLocs.shape[3]))

                dC_squared = torch.matmul(dC_dLocs_xyz, dC_dLocs_xyz.permute(0,2,1))

                # Set bottom right corner to identity
                if num_points_per_cloud is not None:
                    for batch, num_point in enumerate(num_points_per_cloud):
                        ind = torch.arange(num_point, dC_squared.shape[1],device=dC_squared.device)
                        dC_squared[batch][(ind,ind)] = 1
                # Normal inverse
                dampMatrix = self.relaxationFactor*torch.eye(dC_squared.shape[1], device=dC_squared.device)
                dampMatrix = dampMatrix.reshape(1, dC_squared.shape[1], dC_squared.shape[1]).repeat(dC_squared.shape[0],1, 1)
                inv_dC_squared = torch.inverse(dC_squared + dampMatrix)

                inv_dC_squared[torch.isnan(inv_dC_squared)] = 0

                #if torch.any(torch.isnan(inv_dC_squared)):
                #    raise ValueError("Inverting dC_dLocs*dC_dLocs resulted in NaN")


                delta_p_xyz = torch.matmul(inv_dC_squared, C_vector.unsqueeze(-1))
                delta_p_xyz = -torch.matmul(dC_dLocs_xyz.permute(0,2,1), delta_p_xyz)
                delta = delta_p_xyz.reshape(locs.shape)

                s_corr = torch.sum(torch.pow(kernel/self.s_corr_q_kernel, int(self.s_corr_n)).unsqueeze(-1).expand(-1, -1, -1, 3)\
                                  *dSpiky_3_times_D, dim=-2)*self.s_corr_k/self.density_rest
                step = (delta-s_corr)/self.numStaticIterations
                step[torch.isnan(step)] = 0

            else:
                step = 0

            # STATIC COLLISIONS
            for static_iteration in range(self.numStaticIterations):
                locs += step
                if self._apply_collision:
                    locs = self._fix_static_collisions(locs, self.idxs.expand(locs.shape[0], -1),
                                                       self.sdf_pose.expand(locs.shape[0], -1, -1),
                                                       self.scales.expand(locs.shape[0], -1),
                                                       self.collisionDistance)
            if num_points_per_cloud is not None:
                for batch, num_point in enumerate(num_points_per_cloud):
                    locs[batch, num_point:, :] = 0

        return locs

def uniformSource(center_position, initial_velocity, num_particles, batch, max_radius=0.1):
    radius = torch.rand((batch, num_particles)).cuda()*max_radius
    theta  = torch.rand((batch, num_particles)).cuda()*np.pi
    phi    = torch.rand((batch, num_particles)).cuda()*2*np.pi

    pos = torch.empty((batch, num_particles, 3)).cuda()
    pos[:, :, 0] = radius*torch.cos(phi)*torch.sin(theta)
    pos[:, :, 1] = radius*torch.sin(phi)*torch.sin(theta)
    pos[:, :, 2] = radius*torch.cos(theta)

    pos = pos + center_position
    vel = initial_velocity.reshape(1, 1, 3).repeat(batch, num_particles, 1)

    return pos, vel
