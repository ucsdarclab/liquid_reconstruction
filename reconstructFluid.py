import numpy as np
import torch


class ReconstructFluid():
    def __init__(self, gravityForce, fluidConstraint, viscocity, computeNormals, diffRenderer,
                 num_sims, internal_dt, vel_damp=0.1, learning_rate_img = 0.1, learning_rate_particles=150.0,
                 learning_rate_source_loc = 0.25, num_substep_prediction = 1,
                 max_source_radius=0.1, num_fluid_constraints_and_image_loss = 2, img_loss=SmapeLoss(),
                 num_outer_loop = 20, num_img_loss = 2, use_multiple_threads = False, scale_for_cam_render=1,
                 dynamic_addition_of_particles=False, uniform_sample_add_particle = False,
                 use_fluid_prediction = True, min_gradient_to_add_remove_particle=0.20,
                 iou_threshold_to_add_remove_particle = 0.85,
                 emission_rate_scale_for_inserted_particles=0.25):
        '''
        gravityForce: FluidGravityForce object from differentiableFluidSim
        fluidConstraint: MullerConstraints object from differentiableFluidSim
        viscocity:
        computeNormals:
        diffRenderer: PyTorch differentiable renderer that takes in Pointclouds and outputs image
        num_sims: Number of sims ran in parallel to estimate source output rate
        internal_dt: time-step used internal to this object.
        vel_damp:
        learning_rate_img: learning rate from image loss
        learning_rate_particles:
        learning_rate_source_loc:
        max_source_radius: radius for emission source
        num_fluid_constraints_and_image_loss: Number of times to repeat fluid constraints and image loss
        img_loss: loss used for image
        use_multiple_threads: Use multiple threads for differentiable rendering, the rest of the
                              computations are strictly on GPU w/ PyTorch functions, so already pretty fast...
        scale_for_cam_render:
        '''
        self._gravityForce = gravityForce
        self._fluidConstraint = fluidConstraint
        self._diffRenderer = diffRenderer
        self._viscocity = viscocity
        self._computeNormals = computeNormals
        self._num_sims = num_sims
        self._internal_dt = internal_dt
        self._vel_damp = vel_damp
        self._learning_rate_img = learning_rate_img
        self._learning_rate_particles = learning_rate_particles
        self._learning_rate_source_loc = learning_rate_source_loc
        self._max_source_radius = max_source_radius
        self._num_fluid_constraints_and_image_loss = num_fluid_constraints_and_image_loss
        self._num_outer_loop = num_outer_loop
        self._img_loss = img_loss
        self._num_img_loss = num_img_loss
        self._use_multiple_threads = use_multiple_threads
        self._scale_for_cam_render = scale_for_cam_render
        self._num_substep_prediction = num_substep_prediction

        self._use_fluid_prediction = use_fluid_prediction
        self._dynamic_addition_of_particles = dynamic_addition_of_particles
        self._uniform_sample_add_particle = uniform_sample_add_particle
        self._min_gradient_to_add_remove_particle = min_gradient_to_add_remove_particle
        self._iou_threshold_to_add_remove_particle = iou_threshold_to_add_remove_particle
        self._emission_rate_scale_for_inserted_particles = emission_rate_scale_for_inserted_particles
        #self._min_number_particles_to_dynamic_add_remove = min_number_particles_to_dynamic_add_remove

        # DO NOT MESS WITH THIS!! USED FOR OPTIMIAZATION WITH THREADS INTERNALLY
        self._internal_locs_for_gradient_optimization = None
        self._internal_loss_list = None
        self._internal_iou_loss  = None
        self._internal_renders = None
        self._internal_flag_for_dynamic_add_particle = [False for _ in range(num_sims)]


    def uniformSource(self, center_position,  num_particles, batch):
        radius = torch.rand((batch, num_particles), device=center_position.device, dtype=center_position.dtype)*self._max_source_radius
        theta  = torch.rand((batch, num_particles), device=center_position.device, dtype=center_position.dtype)*np.pi
        phi    = torch.rand((batch, num_particles), device=center_position.device, dtype=center_position.dtype)*2*np.pi

        pos = torch.empty((batch, num_particles, 3), device=center_position.device, dtype=center_position.dtype)
        pos[:, :, 0] = radius*torch.cos(phi)*torch.sin(theta)
        pos[:, :, 1] = radius*torch.sin(phi)*torch.sin(theta)
        pos[:, :, 2] = radius*torch.cos(theta)

        pos = pos + center_position
        if num_particles > 0:
            pos = self._fluidConstraint(pos)
        vel = torch.zeros_like(pos)

        return pos, vel

    def get_list_of_particles_to_emit(self, center_num_particles, num_parallel_sims, device):
        '''
        This function is used to generate a list of how many particles to emit.
        E.g.:
            center_num_particles = 0 and num_parallel_sims = 3:
            output: particles_to_emit = [0,1,2]

            center_num_particles = 4 and num_parallel_sims = 3:
            output: particles_to_emit = [3,4,5]
        '''

        # Reset the particles_to_emit so it is "centered" about the current best guess of how many particles need to be emitted
        if center_num_particles <= float(num_parallel_sims)/2.0:
            particles_to_emit = np.arange(num_parallel_sims, dtype=int)
        else:
            particles_to_emit = int(np.around(float(center_num_particles))) \
                                + np.arange(-int(num_parallel_sims/2),-int(num_parallel_sims/2) + num_parallel_sims,
                                            dtype=int)

        return particles_to_emit

    def gradientStepRenderLoss(self, sim_idx, normal, image, occlusion_depth_render = None, render_args = None):

        feature = torch.ones_like(self._internal_locs_for_gradient_optimization[sim_idx])
        self._internal_locs_for_gradient_optimization[sim_idx].requires_grad = True
        particle_pos = self._internal_locs_for_gradient_optimization[sim_idx]*self._scale_for_cam_render
        pcl_int = PointClouds3D(points  = [particle_pos], normals = [normal],features= [feature])

        loss = 0
        has_grad = False
        for cam in range(image.shape[0]):
            i =  cam + sim_idx*image.shape[0]
            if render_args is None:
                # Assumes using DSS so going to render with depth so we can use depth occlusion stuff
                pcl_int = getDepthFeature(pcl_int, self._diffRenderer[cam].cameras)
                result = self._diffRenderer[cam](pcl_int)
                if result is not None:
                    result_depth = result[0, :, :, 0]/self._scale_for_cam_render
                else:
                    result_depth = None
            else:
                # Here we assume PULSAR for differentiable renderer so using forward info for depth
                if occlusion_depth_render is None:
                    result = self._diffRenderer[cam](pcl_int, **render_args)
                    result_depth = None
                else: # Specific for pulsar to handle occlusion
                    result, forward_info = self._diffRenderer[cam](pcl_int, return_forward_info=True, **render_args)
                    result_depth = forward_info[0, :, :, 4]

            # If the rendering is None, then we need to cleanly return out of this function
            #   - No gradient/image updates to the particles
            #   - internal renders for all images are all 0/black colored
            #   - internal loss is computed from all 0/black colored
            if result is None:
                self._internal_locs_for_gradient_optimization[sim_idx].requires_grad = False
                self._internal_loss_list[sim_idx] = 0
                self._internal_iou_loss[sim_idx]  = 0

                for cam in range(image.shape[0]):
                    i =  cam + sim_idx*image.shape[0]
                    self._internal_renders[i, :, :]   = 0
                    self._internal_loss_list[sim_idx] += self._img_loss(image[cam, :, :],
                                                                        self._internal_renders[i, :, :])
                return

            render = result[0, :, :,-1]

            # Mask out from occlusion depth
            if occlusion_depth_render is not None and result_depth is not None:
                render = render * (result_depth < occlusion_depth_render[cam, :, :] )

            self._internal_renders[i, :, :] = render.detach() > 0

            # Get iou loss:
            self._internal_iou_loss[sim_idx] += iouLoss(self._internal_renders[i, :, :], image[cam,:,:])/image.shape[0]

            # Handles case where nothing is visible for rendering
            # We continue here rather than going to "backward" and adding 0 to loss
            # This is because the .backward() pass on the diff-renderer might be unhappy with 0
            # renderable points
            if torch.max(self._internal_renders[i, :, :]) == 0.0:
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
                norm_grad = torch.linalg.norm(self._internal_locs_for_gradient_optimization[sim_idx].grad, dim=-1)
                num_non_zero = torch.sum(norm_grad > 0.0001/self._scale_for_cam_render)

                if num_non_zero == 0:
                    mean_grad = torch.tensor(0, dtype=norm_grad.dtype, device=norm_grad.device)
                else:
                    mean_grad = torch.sum(norm_grad)/num_non_zero

                self._internal_locs_for_gradient_optimization[sim_idx] = self._internal_locs_for_gradient_optimization[sim_idx] \
                                                                         - self._learning_rate_img\
                                                                         *self._internal_locs_for_gradient_optimization[sim_idx].grad
                self._internal_loss_list[sim_idx] = loss.detach()
                print("mean gradient: {}".format(mean_grad))
                print("loss:{}".format(self._internal_iou_loss[sim_idx]))
                #print("loss:{}".format(self._internal_loss_list[sim_idx]))
                self._internal_flag_for_dynamic_add_particle[sim_idx] = (mean_grad < self._min_gradient_to_add_remove_particle) &\
                                                                        (self._iou_threshold_to_add_remove_particle > self._internal_iou_loss[sim_idx])
                                                                        #(0.005 < self._internal_loss_list[sim_idx])
            self._internal_locs_for_gradient_optimization[sim_idx].requires_grad = False

    def nextTimeStep(self, locs, vel, num_particles_emit, source_loc, image,
                     extra_dir_magnitude=None, loc_extra_dir_point_to=None, occlusion_depth_render = None,
                     render_args = None):
        '''
        locs: Nx3 point cloud loc of the fluid from the last timestep
        vel:  Nx3 vel of point cloud of the fluid from the last timestep
        num_particles_emit: floating point which says mean number of particles to emit
        source_loc: emission location
        image: CxWxHx1 images of the new timestep from all the cameras. Note that the number of cameras
                should match with the differentiable renderer. For now, this is just visibility mask!
        extra_dir_magnitude: float which gives magnitude of positional offset given to the particles.
                                This helps with computing more stable normals
        loc_extra_dir_point_to: Vector of size 3  which gives a location the positional offset points towards for
                                every particle.

        '''

        """
        if dt < self._internal_dt:
            raise ValueError('Internal dt is set to {} and timestep called is smaller {}'.format(self._internal_dt, dt))
        """

        initial_num_particles = locs.shape[-2]
        input_locs_list = [ locs.clone() for _ in range(self._num_sims) ]

        # Prepare how many particles to emit:
        num_particles_emit_list = self.get_list_of_particles_to_emit(num_particles_emit, self._num_sims, locs.device)
        num_partitcles_inserted = np.zeros((self._num_sims))
        max_num_particles_emitted = int(np.max(num_particles_emit_list))

        locs = locs.unsqueeze(0)
        vel  = vel.unsqueeze(0)
        for substep_prediction_itr in range(self._num_substep_prediction):
            tmp_loc = locs.clone()
            if self._use_fluid_prediction:
                # Convert locs and vel to Bx(N+newParticles)x3 so we can have multiple going in parallel!
                #   First apply gravity and normal newton equations of motion to the previous particles
                locs, vel = self._gravityForce(locs, vel, self._internal_dt/self._num_substep_prediction)

            if substep_prediction_itr == 0:
                # Also add space according to max number of particles. This is zero padded!
                zero_pad = torch.zeros((1, max_num_particles_emitted, 3), device=locs.device, dtype=locs.dtype)
                locs = torch.cat((locs, zero_pad), dim=-2).repeat(self._num_sims, 1, 1)
                vel  = torch.cat((vel , zero_pad), dim=-2).repeat(self._num_sims, 1, 1)

                # Get new particles from source
                new_locs, new_vel = self.uniformSource(source_loc,
                                                       max_num_particles_emitted,
                                                       self._num_sims)
                # Zero out according to how many particles are supposed to be emitted
                for batch, num_particles_emit_from_list in enumerate(num_particles_emit_list):
                    new_locs[batch, num_particles_emit_from_list:, :] = 0
                    new_vel[ batch, num_particles_emit_from_list:, :] = 0

                # Put particles in to the big zero padded matrix handling all the particles!!
                # Note that this is zero padded
                locs[:, initial_num_particles:, :] = new_locs
                vel [:, initial_num_particles:, :] = new_vel
                num_particles_per_batch = initial_num_particles + num_particles_emit_list

            locs = self._fluidConstraint(locs, num_particles_per_batch)
            vel = (locs[:, :tmp_loc.shape[1], :] - tmp_loc)/(self._internal_dt/self._num_substep_prediction)
            for batch, num_particles_emit_from_list in enumerate(num_particles_emit_list):
                    vel[batch, num_particles_emit_from_list:, :] = 0

        # Now minimize image loss while satisfying fluid constraints
        if self._internal_loss_list is None:
            self._internal_loss_list = torch.zeros((self._num_sims), device=locs.device)

        if self._internal_iou_loss is None:
            self._internal_iou_loss = torch.zeros((self._num_sims), device=locs.device)

        if self._internal_renders is None:
            self._internal_renders = torch.zeros((image.shape[0]*self._num_sims, image.shape[1], image.shape[2]), device=locs.device)



        # Reset internal flags here before starting optimization
        self._internal_flag_for_dynamic_add_particle = [False for _ in range(self._num_sims)]

        for _ in range(self._num_outer_loop):

            # This is the dynamic addition or removal of particles
            #  High level logic is:
            #       -> Look for the particle who satisfies the density constraint the least
            #       -> If the particle's local density is too low, clone that particle
            #       -> If the particle's local density is too high, remove that particle
            #       -> Per parallel sim, the number of particles added/removed is kept track in num_partitcles_inserted
            if self._dynamic_addition_of_particles:

                # This is done with lists for ease of adding, removing particles
                # So using pointclouds 3d to convert between list and padding!
                pcl = PointClouds3D(points=locs)
                pcl._num_points_per_cloud = num_particles_per_batch

                pcl_out_list = []
                for idx, loc_pcl in enumerate(pcl.points_list()):

                    # Only continue if the internal flag is set to be true
                    # This should be managed by other logic (aka from the gradient descent in image loss)\
                    if not self._internal_flag_for_dynamic_add_particle[idx]:
                        pcl_out_list.append(loc_pcl)
                        continue

                    self._internal_flag_for_dynamic_add_particle[idx] = False

                    # If the rendering has a smaller area than the image, then add a particle
                    # Otherwise go to logic for removing particle (else statement)
                    if torch.sum(self._internal_renders[idx*image.shape[0]:idx*image.shape[0]+image.shape[0], :, :]) <\
                            torch.sum(image):
                        print("adding particle")

                        # If using comparision for uniform sampling, then we just randomly pick the index
                        if self._uniform_sample_add_particle:
                            idx_of_particle = np.random.randint(0, high=loc_pcl.shape[0])
                        else:
                            constraint_vector = self._fluidConstraint.evaluateDensityConstraintAfterDuplicatingParticle(loc_pcl.unsqueeze(0))[0]
                            if num_particles_emit_list[idx] == 0:
                                idx_of_particle = torch.argmin(constraint_vector)
                            else:
                                idx_of_particle = torch.argmin(constraint_vector[:-num_particles_emit_list[idx]])

                        add_loc = loc_pcl[idx_of_particle] \
                                  + (torch.rand_like(loc_pcl[idx_of_particle])-0.5)*self._fluidConstraint.radius

                        # Add_loc is added as follows into the loc list:
                        # [ prev_particles, add_loc, particles_from_emission]
                        if num_particles_emit_list[idx] != 0:
                            loc_pcl = torch.cat((loc_pcl[:-num_particles_emit_list[idx]],
                                                 add_loc.unsqueeze(0),
                                                 loc_pcl[-num_particles_emit_list[idx]:]), dim=-2)
                        else:
                            loc_pcl = torch.cat((loc_pcl,
                                                 add_loc.unsqueeze(0)))

                        # The input locs and vels are also updated (so the particle is really cloned)
                        # This clone is duplicated at the end to match the insertion just above^
                        input_locs_list[idx] = torch.cat((input_locs_list[idx], input_locs_list[idx][idx_of_particle].unsqueeze(0)), dim=-2)

                        num_partitcles_inserted[idx] += 1

                    # In this situation we want to remove the particle
                    elif loc_pcl.shape[-2] > 4:
                        print("removing particle {}".format(loc_pcl.shape[-2]))

                        if self._uniform_sample_add_particle:
                            idx_of_particle = np.random.randint(0, high=loc_pcl.shape[0])
                        else:
                            constraint_vector = self._fluidConstraint.evaluateDensityConstraintAfterRemovingParticle(loc_pcl.unsqueeze(0))[0]
                            if num_particles_emit_list[idx] == 0:
                                idx_of_particle = torch.argmin(constraint_vector)
                            else:
                                idx_of_particle = torch.argmin(constraint_vector[:-num_particles_emit_list[idx]])

                        loc_pcl = torch.cat((loc_pcl[:idx_of_particle], loc_pcl[idx_of_particle+1:]), dim=-2)

                        # The input locs and vels are also updated
                        input_locs_list[idx] = torch.cat((input_locs_list[idx][:idx_of_particle],
                                                          input_locs_list[idx][idx_of_particle+1:]), dim=-2)

                        num_partitcles_inserted[idx] -= 1

                    pcl_out_list.append(loc_pcl)
                    num_particles_per_batch[idx] = loc_pcl.shape[0]

                # Use point clouds 3D to convert the lists to padded format!
                pcl = PointClouds3D(pcl_out_list)
                locs = pcl.points_padded()

            for itr_num_fluid_constraints_and_img_loss in range(self._num_fluid_constraints_and_image_loss):

                # Apply fluid constraints
                locs = self._fluidConstraint(locs, num_particles_per_batch)

                # Apply image loss iteratively:
                for _ in range(self._num_img_loss):

                    # Get normals to render
                    pcl = PointClouds3D(points=locs)
                    pcl._num_points_per_cloud = num_particles_per_batch
                    pcl = self._computeNormals(pcl, extra_dir_magnitude, loc_extra_dir_point_to)

                    # Add gradient to locs
                    self._internal_locs_for_gradient_optimization = pcl.points_list()
                    self._internal_loss_list[:] = 0
                    self._internal_iou_loss[:]  = 0

                    # Compute forward rendering for every sim and camera
                    if self._use_multiple_threads:
                        threads = []

                    start_t = time.time()
                    for sim in range(self._num_sims):
                        if self._use_multiple_threads:
                            p = threading.Thread(target=self.gradientStepRenderLoss, args=(sim, pcl.normals_list()[sim],
                                                                                           image, occlusion_depth_render, render_args))
                            p.start()
                            threads.append(p)
                        else:
                            self.gradientStepRenderLoss(sim, pcl.normals_list()[sim], image, occlusion_depth_render, render_args)

                    if self._use_multiple_threads:
                        for p in threads:
                            p.join()

                    # print("Time to finish img loss {}ms".format((time.time() - start_t)*1000.0))

                    pcl = PointClouds3D(points=self._internal_locs_for_gradient_optimization,
                                        features=pcl.features_list(), normals=pcl.normals_list())
                    locs = pcl.points_padded()


        # Update fluid source parameters (location and number of particles emitted)
        # First compute number of particles to insert based on finite difference gradient
        particle_grad = sympy.calculus.finite_diff.apply_finite_diff(1, num_particles_emit_list,
                                                                        self._internal_loss_list.detach().cpu().numpy(),
                                                                        num_particles_emit)

        num_particles_emit = num_particles_emit - particle_grad*self._learning_rate_particles

        particle_idx = np.argmin(np.abs(num_particles_emit_list - num_particles_emit))
        num_particles_emit += self._emission_rate_scale_for_inserted_particles*num_partitcles_inserted[particle_idx] # Increase it by a bit...
        if num_particles_emit < np.min(num_particles_emit_list)-1:
            num_particles_emit = np.min(num_particles_emit_list)-1
        elif num_particles_emit > np.max(num_particles_emit_list)+1:
            num_particles_emit = np.max(num_particles_emit_list) + 1
        if num_particles_emit < 0:
            num_particles_emit = 0

        sim_idx = np.argmin(self._internal_loss_list.detach().cpu().numpy())

        # Output set of locs ( (N + num_particles_emit) x 3):
        locs = locs[sim_idx, :num_particles_per_batch[sim_idx], :]

        # Update source location
        last_index_for_prev = input_locs_list[sim_idx].shape[0]
        if num_particles_emit_list[sim_idx] != 0:
            delta_for_source = torch.mean(locs[last_index_for_prev:, :] \
                                          - new_locs[sim_idx, :num_particles_emit_list[sim_idx]], dim=-2)
            source_loc = source_loc + self._learning_rate_source_loc*delta_for_source

        # Output set of velocity ( (N + num_particles_emit) x 3)
        #vel = vel[sim_idx, :num_particles_per_batch[sim_idx], :]
        vel = torch.zeros_like(locs)
        vel[ :last_index_for_prev, :] = (locs[:last_index_for_prev, :] - input_locs_list[sim_idx]  )/self._internal_dt

        vel = vel*(1 - self._vel_damp)
        vel = self._viscocity(locs.unsqueeze(0), vel.unsqueeze(0)).squeeze(0)

        # Compute density of the result to pass out!
        density_at_particles = self._fluidConstraint.computeDensity(locs.unsqueeze(0)).squeeze(0)

        render_idx = sim_idx*image.shape[0]
        return locs, vel, num_particles_emit, source_loc, self._internal_renders[render_idx:render_idx+image.shape[0], : , :], \
               self._internal_loss_list[sim_idx].detach().cpu().numpy(), self._internal_iou_loss[sim_idx].detach().cpu().numpy(), \
               density_at_particles.detach().cpu().numpy()
