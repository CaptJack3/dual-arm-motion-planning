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

        self.single_mechanical_limit = list(self.ur_params.mechanical_limits.values())[-1][-1]

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

    def sample_random_config(self, goal_prob, goal_conf) -> np.ndarray:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        if np.random.rand() < goal_prob:
            return np.array(goal_conf, dtype=float)
        config = []
        for link in self.ur_params.mechanical_limits.keys():
            low, high = self.ur_params.mechanical_limits[link]
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

        # --- 1. WINDOW CONSTRAINT ---
        # Stack all points once to check the X-limit quickly
        all_centers = np.vstack(list(global_sphere_coords.values()))
        # if np.any(all_centers[:, 0] > 0.4):
        #     return False

        # --- 2. INTERNAL COLLISIONS (Self-Collision) ---
        # Your existing broadcasting implementation was actually good!
        # I just removed the sqrt logic to make it faster.
        for linkA, linkB in self.possible_link_collisions:
            spheres_A = global_sphere_coords[linkA]
            spheres_B = global_sphere_coords[linkB]

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
        # OPTIMIZATION: Vectorized check instead of Triple Loop

        obs_centers = self.env.obstacles  # Shape: (M, 3)
        if len(obs_centers) == 0:
            return True

        obs_r = self.env.radius

        # Iterate over links (only ~6 iterations)
        for link_name, robot_points in global_sphere_coords.items():
            # robot_points shape: (N, 3)
            r_robot = self.ur_params.sphere_radius[link_name]

            # Combined radius squared
            threshold_sq = (r_robot + obs_r) ** 2

            # Broadcast subtraction:
            # (N, 1, 3) - (1, M, 3) = (N, M, 3)
            # This computes distance from EVERY robot sphere to EVERY obstacle sphere instantly
            diff = robot_points[:, None, :] - obs_centers[None, :, :]

            # Squared Euclidean distance
            dist_sq = np.sum(diff * diff, axis=2)  # Shape: (N, M)

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

        # 1. Quick check: If endpoints are invalid, edge is invalid
        # (Usually endpoints are already checked, but RRT* rewiring might skip this)
        # if not self.config_validity_checker(current_conf): return False

        dist = np.linalg.norm(current_conf - prev_conf)
        if dist < 1e-4: return True

        # Calculate number of steps needed
        steps = int(np.ceil(dist / self.resolution))

        # 2. OPTIMIZATION: Bisection Check
        # Instead of linear (0.1, 0.2, 0.3...), check order: 0.5, 0.25, 0.75...
        # This finds collisions faster on average.

        # Generate all alphas linearly first
        alphas = np.linspace(0, 1, steps + 1)[1:-1]  # exclude 0 and 1

        if len(alphas) == 0: return True

        # Sort alphas to mimick bisection (middle first)
        # Simple heuristic: Just shuffle them or pick midpoint
        # For simplicity/speed in numpy, we can just check them.
        # If you really want bisection speed:
        mid = len(alphas) // 2
        # Check middle first
        q_mid = prev_conf + alphas[mid] * (current_conf - prev_conf)
        if not self.config_validity_checker(q_mid): return False

        # Then check the rest
        for alpha in alphas:
            if alpha == alphas[mid]: continue
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

    def validate_IK_solutions(self,configurations : np.array, original_transformation):
        
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
            diff = np.linalg.norm(np.array(original_transformation)-transform_base_to_end)
            if diff < 0.05:
                legal_conf.append(conf)
        if len(legal_conf) == 0:
            raise ValueError("No legal configurations found")       
        return legal_conf
