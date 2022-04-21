import numpy as np
import torch

from pytorch3d.structures.pointclouds import Pointclouds

class SmapeLoss(nn.Module):
    """
    relative L1 norm
    http://drz.disneyresearch.com/~jnovak/publications/KPAL/KPAL.pdf eq(2)
    """
    def __init__(self):
        super(SmapeLoss, self).__init__()

    def forward(self, x, y):
        """
        x pred  (N,3)
        y label (N,3)
        """
        return torch.mean(torch.abs(x-y)/(torch.abs(x)+torch.abs(y)+1e-2))

class ReconstructFluid():
    def __init__(self, gravityForce, fluidConstraint, viscocity, diffRenderer,
                 internal_dt, vel_damp=0.2, learning_rate_img = 0.02, num_substep_prediction = 1,
                 num_outer_loop = 20, num_fluid_constraints_and_image_loss = 2, 
                 img_loss=SmapeLoss(), num_img_loss = 5, scale_for_cam_render=1,
                 min_gradient_to_add_remove_particle=0.001,
                 iou_threshold_to_add_remove_particle = 0.9):
        '''
        gravityForce: FluidGravityForce object from differentiableFluidSim
        fluidConstraint: MullerConstraints object from differentiableFluidSim
        viscocity: XsphViscosity object from differentiableFluidSim 
        diffRenderer: PyTorch differentiable renderer that takes in Pointclouds and outputs image
        internal_dt: time-step between each
        vel_damp: dampening factor of particle velocity
        learning_rate_img: learning rate from image loss
        num_substep_prediction: Number of substeps to predict particles with
        num_outer_loop: Number of times to repeat entire optimizations steps
        num_fluid_constraints_and_image_loss: Number of times to repeat fluid constraints and image loss
        img_loss: loss used for image
        num_img_loss: number of iterations for image loss
        scale_for_cam_render: Scales particle positions before rendering them
        min_gradient_to_add_remove_particle: minimum gradient to add/remove particles
        iou_threshold_to_add_remove_particle: maximum iou to add/remove particles
        '''
        self._gravityForce = gravityForce
        self._fluidConstraint = fluidConstraint
        self._diffRenderer = diffRenderer
        self._viscocity = viscocity
        self._internal_dt = internal_dt
        self._vel_damp = vel_damp
        self._learning_rate_img = learning_rate_img
        self._num_fluid_constraints_and_image_loss = num_fluid_constraints_and_image_loss
        self._num_outer_loop = num_outer_loop
        self._img_loss = img_loss
        self._num_img_loss = num_img_loss
        self._use_multiple_threads = use_multiple_threads
        self._scale_for_cam_render = scale_for_cam_render
        self._num_substep_prediction = num_substep_prediction
        self._min_gradient_to_add_remove_particle = min_gradient_to_add_remove_particle
        self._iou_threshold_to_add_remove_particle = iou_threshold_to_add_remove_particle

        # DO NOT MESS WITH THIS!! USED FOR OPTIMIAZATION WITH THREADS INTERNALLY
        self._internal_locs_for_gradient_optimization = None
        self._internal_loss_list = None
        self._internal_iou_loss  = None
        self._internal_renders = None
        self._internal_flag_for_dynamic_add_particle = [False for _ in range(num_sims)]

    def gradientStepRenderLoss(self, image, occlusion_depth_render = None, render_args = None):

        feature = torch.ones_like(self._internal_locs_for_gradient_optimization)
        self._internal_locs_for_gradient_optimization.requires_grad = True
        particle_pos = self._internal_locs_for_gradient_optimization*self._scale_for_cam_render
        pcl_int = Pointclouds(points  = particle_pos, features= feature)

        loss = 0
        has_grad = False
        for cam in range(image.shape[0]):
            # Here we assume PULSAR for differentiable renderer so using forward info for depth
            if occlusion_depth_render is None:
                result = self._diffRenderer[cam](pcl_int, **render_args)
                result_depth = None
            else: # Specific for pulsar to handle occlusion
                result, forward_info = self._diffRenderer[cam](pcl_int, return_forward_info=True, **render_args)
                result_depth = forward_info[0, :, :, 4]

            # If the rendering is None then skip this rendering for our loss back-propogatin
            if result is None:
                # self._internal_locs_for_gradient_optimization.requires_grad = False
                # self._internal_loss_list = 0
                # self._internal_iou_loss  = 0
                # self._internal_renders[cam, :, :]   = 0
                continue

            render = result[0, :, :,-1]

            # Mask out from occlusion depth
            if occlusion_depth_render is not None and result_depth is not None:
                render = render * (result_depth < occlusion_depth_render[cam, :, :] )

            self._internal_renders[cam, :, :] = render.detach() > 0

            # Get iou loss:
            self._internal_iou_loss += iouLoss(self._internal_renders[cam, :, :], image[cam,:,:])/image.shape[0]

            # Handles case where nothing is visible for rendering
            # We continue here rather than going to "backward" and adding 0 to loss
            # This is because the .backward() pass on the diff-renderer might be unhappy with 0
            # renderable points
            if torch.max(self._internal_renders[cam, :, :]) == 0.0:
                continue

            has_grad = True
            loss += self._img_loss(image[cam, :, :], render)

        # Back-propogate loss:
        if has_grad:
            try:
                loss.backward()
            except Exception as e:
                has_grad = False


        with torch.no_grad():
            if has_grad:
                norm_grad = torch.linalg.norm(self._internal_locs_for_gradient_optimization.grad, dim=-1)
                num_non_zero = torch.sum(norm_grad > 0.0001/self._scale_for_cam_render)

                if num_non_zero == 0:
                    mean_grad = torch.tensor(0, dtype=norm_grad.dtype, device=norm_grad.device)
                else:
                    mean_grad = torch.sum(norm_grad)/num_non_zero

                self._internal_locs_for_gradient_optimization = self._internal_locs_for_gradient_optimization \
                                                                    - self._learning_rate_img\
                                                                    *self._internal_locs_for_gradient_optimization.grad
                self._internal_loss_list = loss.detach()
                print("mean gradient: {}".format(mean_grad))
                print("loss:{}".format(self._internal_iou_loss))
                #print("loss:{}".format(self._internal_loss_list[sim_idx]))
                self._internal_flag_for_dynamic_add_particle = (mean_grad < self._min_gradient_to_add_remove_particle) &\
                                                                        (self._iou_threshold_to_add_remove_particle > self._internal_iou_loss)
            self._internal_locs_for_gradient_optimization.requires_grad = False

    def nextTimeStep(self, locs, vel, image, occlusion_depth_render = None,
                     render_args = None):
        '''
        locs: Nx3 point cloud loc of the fluid from the last timestep
        vel:  Nx3 vel of point cloud of the fluid from the last timestep
        image: CxWxHx1 images of the new timestep from all the cameras. Note that the number of cameras
                should match with the differentiable renderer. For now, this is just visibility mask!
        occlusion_depth_render: CxWxHx1 depth rendering to do occlusion
        '''

        locs = locs.unsqueeze(0)
        vel  = vel.unsqueeze(0)
        input_locs = locs.clone()

        # Predict and solve initial fliud constriants to keep stuff stable
        for substep_prediction_itr in range(self._num_substep_prediction):
            tmp_loc = locs.clone()

            locs, vel = self._gravityForce(locs, vel, self._internal_dt/self._num_substep_prediction)
            locs = self._fluidConstraint(locs)
            vel = (locs[:, :tmp_loc.shape[1], :] - tmp_loc)/(self._internal_dt/self._num_substep_prediction)

        # Reset internal flags here before starting optimization
        self._internal_flag_for_dynamic_add_particle = False

        for _ in range(self._num_outer_loop):

            # This is the dynamic addition or removal of particlesk
            # Note that self._internal_flag_for_dynamic_add_particle is set when 
            # minimizing image loss (i.e. in function gradientStepRenderLoss)
            if self._internal_flag_for_dynamic_add_particle:
                self._internal_flag_for_dynamic_add_particle = False

                # If the rendering has a smaller area than the image, then add a particle
                # Otherwise go to logic for removing particle (else statement)
                if torch.sum(self._internal_renders) < torch.sum(image):
                    print("adding particle")

                    # Find particle that would best satisify the density constraint after duplicating
                    constraint_vector = self._fluidConstraint.evaluateDensityConstraintAfterDuplicatingParticle(locs)[0]
                    idx_of_particle = torch.argmin(constraint_vector)

                    # Add a bit of RNG noise so the duplication is not exactly at the same location
                    add_loc = locs[:, idx_of_particle] \
                                + (torch.rand_like(locs[:,idx_of_particle])-0.5)*self._fluidConstraint.radius
                    locs = torch.cat((locs, add_loc), dim=-2)
                    input_locs = torch.cat((input_locs, input_locs[:, idx_of_particle]), dim=-2)

                # In this situation we want to remove the particle
                elif locs.shape[-2] > 4:
                    print("Removing particle")

                    # Find particle that would best satisify the density constraint after removing
                    constraint_vector = self._fluidConstraint.evaluateDensityConstraintAfterRemovingParticle(locs)[0]
                    idx_of_particle = torch.argmin(constraint_vector)

                    # Remove particle
                    locs = torch.cat((locs[:, :idx_of_particle], 
                                      locs[:, idx_of_particle+1:]), dim=-2)
                    input_locs = torch.cat((input_locs[:, :idx_of_particle], 
                                            input_locs[:, idx_of_particle+1:]), dim=-2)

            # Minimize image loss while satisfying constriants
            for itr_num_fluid_constraints_and_img_loss in range(self._num_fluid_constraints_and_image_loss):

                # Apply fluid constraints
                locs = self._fluidConstraint(locs)

                # Apply image loss iteratively:
                for _ in range(self._num_img_loss):

                    # Add gradient to locs
                    self._internal_locs_for_gradient_optimization = locs
                    self._internal_loss_list = 0
                    self._internal_iou_loss  = 0
                    self._internal_renders = torch.zeros_like(image)

                    self.gradientStepRenderLoss(sim, pcl.normals_list()[sim], image, occlusion_depth_render, render_args)

                    locs = self._internal_locs_for_gradient_optimization


        # Compute velocities
        vel = torch.zeros_like(locs)
        vel[ :last_index_for_prev, :] = (locs - input_locs)/self._internal_dt

        vel = vel*(1 - self._vel_damp)
        vel = self._viscocity(locs, vel)

        # Compute density of the result to pass out!
        density_at_particles = self._fluidConstraint.computeDensity(locs).squeeze(0)

        return locs.squeeze(0), vel.squeeze(0), \
               self._internal_renders.detach() \
               self._internal_loss_list.detach(), 
               self._internal_iou_loss.detach(), \
               density_at_particles
