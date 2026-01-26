import numpy as np
import inverse_kinematics
from kinematics import Transform, UR5e_PARAMS, UR5e_without_camera_PARAMS

class BuildingBlocks3D(object):
    """
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    """

    def __init__(self, transform:Transform, ur_params:UR5e_PARAMS, env, resolution=0.1):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution

        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechamical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [
            ["shoulder_link", "forearm_link"],
            ["shoulder_link", "wrist_1_link"],
            ["shoulder_link", "wrist_2_link"],
            ["shoulder_link", "wrist_3_link"],
            ["upper_arm_link", "wrist_1_link"],
            ["upper_arm_link", "wrist_2_link"],
            ["upper_arm_link", "wrist_3_link"],
            ["forearm_link", "wrist_2_link"],
            ["forearm_link", "wrist_3_link"],
        ]

    def _to_xyz_arrays(self, global_sphere_coords):
        """
        Make Transform.conf2sphere_coords output consistent:
        - Old HW: dict(link -> np.ndarray (N,3))
        - Current HW: dict(link -> list of homogeneous (4,) points in WORLD frame)
        Returns: dict(link -> np.ndarray (N,3)) in WORLD frame.
        """
        fixed = {}
        for link, pts in global_sphere_coords.items():
            arr = np.asarray(pts, dtype=float)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[1] >= 4:
                arr = arr[:, :3]
            fixed[link] = arr
        return fixed

    def edge_cost(self, conf1, conf2):
        """Alias used by some planners/trees."""
        conf1 = np.asarray(conf1, dtype=float)
        conf2 = np.asarray(conf2, dtype=float)
        return self.compute_distance(conf1, conf2)

    def sample_random_config(self, goal_prob, goal_conf) -> np.ndarray:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        if np.random.rand() < goal_prob:
            return np.array(goal_conf, dtype=float)

        config = []
        # NOTE: current kinematics uses 'mechamical_limits' (typo in params)
        for low, high in self.ur_params.mechamical_limits.values():
            config.append(np.random.uniform(low, high))
        return np.array(config, dtype=float)

    def config_validity_checker_ORIG(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return False if in collision
        @param conf - some configuration
        """
        # TODO: HW2 5.2.2- Pay attention that function is a little different than in HW2
        pass

    def config_validity_checker(self, conf) -> bool:
        """
        Optimized collision checker:
        1. Uses Squared Norm to avoid expensive Sqrt operations.
        2. Vectorizes the Robot-vs-Obstacles loop to eliminate Python overhead.
        """
        # Forward Kinematics (Unavoidable overhead)
        global_sphere_coords = self.transform.conf2sphere_coords(conf)

        # Convert kinematics output (lists of homogeneous points) -> np arrays (N,3)
        global_sphere_coords = self._to_xyz_arrays(global_sphere_coords)

        # --- 1. WORKSPACE / BBOX CONSTRAINT (optional) ---
        # If environment defines a bbox, reject configs that go outside it.
        # env.bbox is of the form [[xmin, ymax], [xmax, ymin]]
        if hasattr(self.env, "bbox") and self.env.bbox is not None:
            xmin = float(self.env.bbox[0][0])
            xmax = float(self.env.bbox[1][0])
            ymin = float(self.env.bbox[1][1])
            ymax = float(self.env.bbox[0][1])

            all_centers = np.vstack(list(global_sphere_coords.values()))
            xs = all_centers[:, 0]
            ys = all_centers[:, 1]
            if np.any(xs < xmin) or np.any(xs > xmax) or np.any(ys < ymin) or np.any(ys > ymax):
                return False

        # --- 2. INTERNAL COLLISIONS (Self-Collision) ---
        for linkA, linkB in self.possible_link_collisions:
            spheres_A = global_sphere_coords[linkA]  # (N,3)
            spheres_B = global_sphere_coords[linkB]  # (M,3)

            # Precompute squared threshold
            rA = self.ur_params.sphere_radius[linkA]
            rB = self.ur_params.sphere_radius[linkB]
            rad_sum_sq = (rA + rB) ** 2

            # Broadcasting: (N, 1, 3) - (1, M, 3) -> (N, M, 3)
            diff = spheres_A[:, None, :] - spheres_B[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)

            if np.any(dist_sq < rad_sum_sq):
                return False

        # --- 3. EXTERNAL COLLISIONS (Robot vs Obstacles) ---
        obs_centers = self.env.obstacles  # expected shape: (M, 3)
        if obs_centers is None:
            obs_centers = np.zeros((0, 3), dtype=float)
        else:
            obs_centers = np.asarray(obs_centers, dtype=float)
            if obs_centers.size == 0:
                obs_centers = np.zeros((0, 3), dtype=float)
            elif obs_centers.ndim == 1:
                obs_centers = obs_centers.reshape(1, -1)
            # Drop homogeneous coord if exists
            if obs_centers.shape[1] >= 4:
                obs_centers = obs_centers[:, :3]

        if obs_centers.shape[0] == 0:
            return True

        obs_r = self.env.radius

        # Iterate over links (only ~6 iterations)
        for link_name, robot_points in global_sphere_coords.items():
            # robot_points shape: (N, 3)
            r_robot = self.ur_params.sphere_radius[link_name]

            # Combined radius squared
            threshold_sq = (r_robot + obs_r) ** 2

            # Broadcast subtraction: (N,1,3) - (1,M,3) => (N,M,3)
            diff = robot_points[:, None, :] - obs_centers[None, :, :]

            # Squared Euclidean distance
            dist_sq = np.sum(diff * diff, axis=2)  # (N, M)

            if np.any(dist_sq < threshold_sq):
                return False

        return True

    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        """
        Optimized edge checker:
        1. Checks endpoints first.
        2. Checks midpoint, then quarters (Bisection) for faster fail detection.
        """
        prev_conf = np.array(prev_conf, dtype=float)
        current_conf = np.array(current_conf, dtype=float)

        dist = np.linalg.norm(current_conf - prev_conf)
        if dist < 1e-4:
            return True

        # Calculate number of steps needed
        steps = int(np.ceil(dist / self.resolution))

        # Generate all alphas linearly first
        alphas = np.linspace(0, 1, steps + 1)[1:-1]  # exclude 0 and 1
        if len(alphas) == 0:
            return True

        # Check middle first
        mid = len(alphas) // 2
        q_mid = prev_conf + alphas[mid] * (current_conf - prev_conf)
        if not self.config_validity_checker(q_mid):
            return False

        # Then check the rest
        for alpha in alphas:
            if alpha == alphas[mid]:
                continue
            q = prev_conf + alpha * (current_conf - prev_conf)
            if not self.config_validity_checker(q):
                return False

        return True

    def compute_distance(self, conf1, conf2):
        """
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        """
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5

    def validate_IK_solutions(self, configurations : np.array, original_transformation):
        legal_conf = []
        limits = list(self.ur_params.mechamical_limits.values())

        for conf in configurations:
            # check for angles limits
            valid_angles = True
            for i, angle in enumerate(conf):
                if not (limits[i][0] <= angle <= limits[i][1]):
                    valid_angles = False
                    break
            if not valid_angles:
                continue

            # check for collision
            if not self.config_validity_checker(conf):
                continue

            # verify solution: make the difference between the solution and the original matrix and calculate the norm
            transform_base_to_end = inverse_kinematics.forward_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, conf)
            diff = np.linalg.norm(np.array(original_transformation) - transform_base_to_end)
            if diff < 0.05:
                legal_conf.append(conf)

        if len(legal_conf) == 0:
            raise ValueError("No legal configurations found")

        return legal_conf
