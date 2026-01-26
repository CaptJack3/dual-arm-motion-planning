import json
import time
from enum import Enum
import numpy as np
from matplotlib.sankey import RIGHT

from environment import Environment

from kinematics import UR5e_PARAMS, UR5e_without_camera_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import BuildingBlocks3D
from visualizer import Visualize_UR

import inverse_kinematics

from environment import LocationType


def log(msg):
    written_log = f"STEP: {msg}"
    print(written_log)
    dir_path = r"./outputs/"
    with open(dir_path + 'output.txt', 'a') as file:
        file.write(f"{written_log}\n")


class Gripper(str, Enum):
    OPEN = "OPEN",
    CLOSE = "CLOSE"
    STAY = "STAY"




def update_environment(env, active_arm, static_arm_conf, cubes_positions):
    env.set_active_arm(active_arm)
    env.update_obstacles(cubes_positions, static_arm_conf)


class Experiment:
    def __init__(self, cubes=None):
        # environment params
        self.cubes = cubes
        self.right_arm_base_delta = 0  # deviation from the first link 0 angle
        self.left_arm_base_delta = 0  # deviation from the first link 0 angle
        self.right_arm_meeting_safety = None
        self.left_arm_meeting_safety = None

        # tunable params
        self.max_itr = 1000
        self.max_step_size = 0.5
        self.goal_bias = 0.05
        self.resolution = 0.1
        # start confs
        self.right_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        self.left_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        # result dict
        self.experiment_result = []

    def push_step_info_into_single_cube_passing_data(self, description, active_id, command, static_conf, path, cubes, gripper_pre, gripper_post):
        self.experiment_result[-1]["description"].append(description)
        self.experiment_result[-1]["active_id"].append(active_id)
        self.experiment_result[-1]["command"].append(command)
        self.experiment_result[-1]["static"].append(static_conf)
        self.experiment_result[-1]["path"].append(path)
        self.experiment_result[-1]["cubes"].append(cubes)
        self.experiment_result[-1]["gripper_pre"].append(gripper_pre)
        self.experiment_result[-1]["gripper_post"].append(gripper_post)

    def plan_single_arm(self, planner, start_conf, goal_conf, description, active_id, command, static_arm_conf, cubes_real,
                            gripper_pre, gripper_post):
        path, cost = planner.find_path(start_conf=start_conf,
                                       goal_conf=goal_conf,
                                       manipulator=active_id)
        # create the arm plan
        self.push_step_info_into_single_cube_passing_data(description,
                                                          active_id,
                                                          command,
                                                          static_arm_conf.tolist(),
                                                          [path_element.tolist() for path_element in path],
                                                          [list(cube) for cube in cubes_real],
                                                          gripper_pre,
                                                          gripper_post)

    def plan_single_cube_passing(self, cube_i, cubes,
                                 left_arm_start, right_arm_start,
                                 env, bb, planner, left_arm_transform, right_arm_transform,):
        # add a new step entry
        single_cube_passing_info = {
            "description": [],  # text to be displayed on the animation
            "active_id": [],  # active arm id
            "command": [],
            "static": [],  # static arm conf
            "path": [],  # active arm path
            "cubes": [],  # coordinates of cubes on the board at the given timestep
            "gripper_pre": [],  # active arm pre path gripper action (OPEN/CLOSE/STAY)
            "gripper_post": []  # active arm pre path gripper action (OPEN/CLOSE/STAY)
        }
        self.experiment_result.append(single_cube_passing_info)
        ###############################################################################
        # set variables
        description = "right_arm => [start -> cube pickup], left_arm static"
        active_arm = LocationType.RIGHT
        # start planning
        log(msg=description)
    
      
        #################################################################################
        #                                                                               #
        #   #######  #######      ######   #######      #######                         #
        #      #     #     #      #     #  #     #     #       #                        #
        #      #     #     #      #     #  #     #             #                        #
        #      #     #     #      #     #  #     #       ######                         #
        #      #     #     #      #     #  #     #      #                               #
        #      #     #     #      #     #  #     #      #                               #
        #      #     #######      ######   #######      ########                        #
        #                                                                               #
        #################################################################################
        ''' cube_approach = None #TODO 2: find a conf for the arm to get the correct cube '''

        # ============================
        # TODO 2: find cube_approach (RIGHT arm config that is above cube_i)
        # ============================

        # Active arm is RIGHT, left arm is static at left_arm_start during this planning segment
        update_environment(env, LocationType.RIGHT, left_arm_start, cubes)

        cube_pos = np.array(cubes[cube_i], dtype=float)  # [x,y,z] in world frame
        cube_side = 0.04  # as in get_cubes_for_experiment()
        above_offset = 0.12  # "approach height" above cube center (tune if needed)

        # target tool position: above cube
        p_approach = cube_pos + np.array([0.0, 0.0, above_offset], dtype=float)

        # tool orientation: "down-ish" (same convention you use for meeting)
        rpy_pick = [np.pi, 0.0, 0.0]

        # Build target pose and solve IK
        T_pick = right_arm_transform.get_base_to_tool_transform(position=p_approach, rpy=rpy_pick)
        IK = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_pick)

        # Validate IK solutions (joint limits + collisions + FK tolerance)
        valid = bb.validate_IK_solutions(IK, T_pick)
        if len(valid) == 0:
            raise RuntimeError(
                f"TODO2 failed: no valid IK for cube approach (cube {cube_i}). "
                "Try changing above_offset or rpy_pick."
            )

        # Choose the IK solution closest to the current right arm start
        cube_approach = min(valid, key=lambda q: np.linalg.norm(q - right_arm_start))

        #TODO  2 - END

        # plan the path
        self.plan_single_arm(planner, right_arm_start, cube_approach, description, active_arm, "move",
                                 left_arm_start, cubes, Gripper.OPEN, Gripper.STAY)
        ###############################################################################

        # self.push_step_info_into_single_cube_passing_data("picking up a cube: go down",
        #                                                   LocationType.RIGHT,
        #                                                   "movel",
        #                                                   list(self.left_arm_home),
        #                                                   [0, 0, -0.14],
        #                                                   [],
        #                                                   Gripper.STAY,
        #                                                   Gripper.CLOSE)

        #TODO - Check if it was needed to add the cubes into []
        # TODO - check is this left_arm_start.tolist() is better than list(self.left_arm_home)

        self.push_step_info_into_single_cube_passing_data("picking up a cube: go down",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          left_arm_start.tolist(),
                                                          [0, 0, -0.14],
                                                          [list(cube) for cube in cubes],
                                                          Gripper.STAY,
                                                          Gripper.CLOSE)


        #################################################################################
        #                                                                               #
        #   #######  #######      ######   #######      #######                         #
        #      #     #     #      #     #  #     #     #       #                        #
        #      #     #     #      #     #  #     #             #                        #
        #      #     #     #      #     #  #     #       ######                         #
        #      #     #     #      #     #  #     #             #                        #
        #      #     #     #      #     #  #     #     #       #                        #
        #      #     #######      ######   #######      #######                         #
        #                                                                               #
        #################################################################################
        #return None, None #TODO 3: return left and right end position, so it can be the start position for the next interation.

        # ============================
        # TODO 3: complete the cube passing (A -> meeting -> B)
        # ============================

        # Keep cube list formatting consistent
        cubes_ll = [list(cube) for cube in cubes]

        # (A) Retract up after grasp
        self.push_step_info_into_single_cube_passing_data(
            "picking up a cube: go up",
            LocationType.RIGHT,
            "movel",
            left_arm_start.tolist(),
            [0, 0, 0.14],
            cubes_ll,
            Gripper.STAY,
            Gripper.STAY
        )

        # Optional but useful: treat the carried cube as removed from obstacles while moving
        cubes_wo_i = [cubes[j] for j in range(len(cubes)) if j != cube_i]
        cubes_wo_i_ll = [list(cube) for cube in cubes_wo_i]

        # (B) RIGHT: go from cube_approach -> right meeting conf
        description = "right_arm => [cube pickup -> meeting], left_arm static"
        log(msg=description)
        update_environment(env, LocationType.RIGHT, left_arm_start, cubes_wo_i)
        self.plan_single_arm(planner, cube_approach, self.right_arm_meeting_conf, description,
                             LocationType.RIGHT, "move",
                             left_arm_start, cubes_wo_i, Gripper.STAY, Gripper.STAY)
        right_at_meeting = np.array(self.right_arm_meeting_conf, dtype=float)

        # (C) LEFT: go from left_arm_start -> left meeting conf
        description = "left_arm => [start -> meeting], right_arm static"
        log(msg=description)
        update_environment(env, LocationType.LEFT, right_at_meeting, cubes_wo_i)
        self.plan_single_arm(planner, left_arm_start, self.left_arm_meeting_conf, description,
                             LocationType.LEFT, "move",
                             right_at_meeting, cubes_wo_i, Gripper.STAY, Gripper.OPEN)
        left_at_meeting = np.array(self.left_arm_meeting_conf, dtype=float)

        # (D) Handover: LEFT closes, RIGHT opens (using small approach/retreat movel)
        # You can tune these small numbers later.
        self.push_step_info_into_single_cube_passing_data(
            "handover: right arm approaches",
            LocationType.RIGHT,
            "movel",
            left_at_meeting.tolist(),
            [-0.06, 0, 0],
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.STAY
        )
        self.push_step_info_into_single_cube_passing_data(
            "handover: left arm approaches + grasps",
            LocationType.LEFT,
            "movel",
            right_at_meeting.tolist(),
            [-0.06, 0, 0],
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.CLOSE
        )
        self.push_step_info_into_single_cube_passing_data(
            "handover: right releases + retreats",
            LocationType.RIGHT,
            "movel",
            left_at_meeting.tolist(),
            [0.06, 0, 0],
            cubes_wo_i_ll,
            Gripper.OPEN,
            Gripper.STAY
        )
        self.push_step_info_into_single_cube_passing_data(
            "handover: left retreats",
            LocationType.LEFT,
            "movel",
            right_at_meeting.tolist(),
            [0.06, 0, 0],
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.STAY
        )

        # (E) Compute LEFT "place approach" in Zone B using IK (mirror cube position from RIGHT zone to LEFT zone)
        # Map the cube's local offset in the right area into the left area.
        right_corner = np.array(env.cube_area_corner[LocationType.RIGHT], dtype=float)
        left_corner = np.array(env.cube_area_corner[LocationType.LEFT], dtype=float)

        cube_pos = np.array(cubes[cube_i], dtype=float)
        local_offset = cube_pos - right_corner
        place_cube_pos = left_corner + local_offset

        cube_side = 0.04
        place_cube_pos[2] = cube_side / 2.0  # cube sits on table

        above_place = 0.25
        p_place_approach = place_cube_pos + np.array([0.0, 0.0, above_place], dtype=float)
        rpy_place = [np.pi, 0.0, 0.0]

        update_environment(env, LocationType.LEFT, right_at_meeting, cubes_wo_i)
        T_place = left_arm_transform.get_base_to_tool_transform(position=p_place_approach, rpy=rpy_place)
        IKp = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_place)
        validp = bb.validate_IK_solutions(IKp, T_place)
        if len(validp) == 0:
            raise RuntimeError("TODO3: no valid IK for left place-approach. Try changing above_place or rpy_place.")

        left_place_approach = min(validp, key=lambda q: np.linalg.norm(q - left_at_meeting))

        # (F) LEFT: meeting -> place approach (RRT*)
        description = "left_arm => [meeting -> place approach], right_arm static"
        log(msg=description)
        update_environment(env, LocationType.LEFT, right_at_meeting, cubes_wo_i)
        self.plan_single_arm(planner, left_at_meeting, left_place_approach, description,
                             LocationType.LEFT, "move",
                             right_at_meeting, cubes_wo_i, Gripper.STAY, Gripper.STAY)

        # (G) Place: down + open, then up
        self.push_step_info_into_single_cube_passing_data(
            "dropping a cube: go down",
            LocationType.LEFT,
            "movel",
            right_at_meeting.tolist(),
            [0, 0, -0.25],
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.OPEN
        )
        self.push_step_info_into_single_cube_passing_data(
            "dropping a cube: go up",
            LocationType.LEFT,
            "movel",
            right_at_meeting.tolist(),
            [0, 0, 0.25],
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.STAY
        )

        # Update cube position in the shared list so future iterations see it on the left side
        cubes[cube_i] = place_cube_pos.tolist()

        left_arm_end = np.array(left_place_approach, dtype=float)
        right_arm_end = right_at_meeting
        return left_arm_end, right_arm_end


    def plan_experiment(self):
        start_time = time.time()

        exp_id = 2
        ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
        ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)

        env = Environment(ur_params=ur_params_right)

        transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT])
        transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT])

        env.arm_transforms[LocationType.RIGHT] = transform_right_arm
        env.arm_transforms[LocationType.LEFT] = transform_left_arm
        '''
        bb = BuildingBlocks3D(env=env,
                             resolution=self.resolution,
                             p_bias=self.goal_bias, )

        rrt_star_planner = RRT_STAR(max_step_size=self.max_step_size,
                                    max_itr=self.max_itr,
                                    bb=bb)
        '''
        # Building blocks: one per arm (different transform + params)
        bb_right = BuildingBlocks3D(transform=transform_right_arm,
                                    ur_params=ur_params_right,
                                    env=env,
                                    resolution=self.resolution)

        bb_left = BuildingBlocks3D(transform=transform_left_arm,
                                   ur_params=ur_params_left,
                                   env=env,
                                   resolution=self.resolution)

        # Planners: one per arm (cleanest)
        planner_right = RRT_STAR(max_step_size=self.max_step_size,
                                 max_itr=self.max_itr,
                                 bb=bb_right)

        planner_left = RRT_STAR(max_step_size=self.max_step_size,
                                max_itr=self.max_itr,
                                bb=bb_left)

        visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                                  transform_left_arm=transform_left_arm)
        # cubes
        if self.cubes is None:
            self.cubes = self.get_cubes_for_experiment(exp_id, env)

        log(msg="calculate meeting point for the test.")
        ################################################################################
        #                                                                               #
        #   #######  #######      ######   #######        #                             #
        #      #     #     #      #     #  #     #       ##                             #
        #      #     #     #      #     #  #     #      # #                             #
        #      #     #     #      #     #  #     #        #                             #
        #      #     #     #      #     #  #     #        #                             #
        #      #     #     #      #     #  #     #        #                             #
        #      #     #######      ######   #######      #####                           #
        #                                                                               #
        #################################################################################

        # ============================
        # TODO 1: meeting confs (deterministic)
        #   - M.x,M.y from env midpoint
        #   - z = 0.5 fixed
        #   - delta = 0.05 fixed
        #   - Right/Left orientations differ by 90 deg yaw (cube face pair swap)
        # ============================

        bL = np.array(env.arm_base_location[LocationType.LEFT], dtype=float)
        bR = np.array(env.arm_base_location[LocationType.RIGHT], dtype=float)

        # Meeting point (midpoint in XY, fixed Z)
        M = np.array([(bL[0] + bR[0]) / 2.0,
                      (bL[1] + bR[1]) / 2.0,
                      0.50], dtype=float)

        # Separation direction between bases
        u = (bL - bR)
        u = u / np.linalg.norm(u)

        delta = 0.05  # 5 cm (fixed)
        delta_vec = delta * u

        pR = M - delta_vec
        pL = M + delta_vec

        # ----------------------------
        # Orientation design:
        # Right pose = base meeting orientation
        # Left pose  = rotated by +90deg around world Z (yaw) to grasp different cube faces
        # ----------------------------
        # Choose a baseline RPY for the right tool.
        # If your tool frame is different, you might need to tweak roll/pitch.
        rpy_right = [np.pi, 0.0, 0.0]

        # "Rotate 90 degrees" about Z axis => yaw += pi/2
        rpy_left = [rpy_right[0], rpy_right[1], rpy_right[2] + np.pi / 2]

        # ---- RIGHT ARM IK ----
        update_environment(env, LocationType.RIGHT, self.left_arm_home, self.cubes)
        T_R = transform_right_arm.get_base_to_tool_transform(position=pR, rpy=rpy_right)
        IK_R = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_R)
        valid_R = bb.validate_IK_solutions(IK_R, T_R)
        if len(valid_R) == 0:
            raise RuntimeError("TODO1: No valid RIGHT meeting IK solutions. Try changing rpy_right.")

        # Pick solution closest to home
        self.right_arm_meeting_conf = min(valid_R, key=lambda q: np.linalg.norm(q - self.right_arm_home)) # TODO 1

        # ---- LEFT ARM IK ----
        update_environment(env, LocationType.LEFT, self.right_arm_home, self.cubes)
        T_L = transform_left_arm.get_base_to_tool_transform(position=pL, rpy=rpy_left)
        IK_L = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_L)
        valid_L = bb.validate_IK_solutions(IK_L, T_L)
        if len(valid_L) == 0:
            raise RuntimeError(
                "TODO1: No valid LEFT meeting IK solutions. Try changing rpy_left (or rpy_right baseline).")

        self.left_arm_meeting_conf = min(valid_L, key=lambda q: np.linalg.norm(q - self.left_arm_home))  #TODO 1

        log(msg=f"Meeting M = {M.tolist()}, delta={delta}")
        log(msg=f"Right meeting target pR = {pR.tolist()}, rpy_right={rpy_right}")
        log(msg=f"Left  meeting target pL = {pL.tolist()}, rpy_left={rpy_left}")



        ################################################################################
        #                                                                               #
        #   #######  #######      ######   #######        #                             #
        #      #     #     #      #     #  #     #       ##                             #
        #      #     #     #      #     #  #     #      # #                             #
        #      #     #     #      #     #  #     #        #                             #
        #      #     #     #      #     #  #     #        #                             #
        #      #     #     #      #     #  #     #        #                             #
        #      #     #######      ######   #######      #####                           #
        #                                                                               #
        #################################################################################

        log(msg="start planning the experiment.")
        left_arm_start = self.left_arm_home
        right_arm_start = self.right_arm_home
        for i in range(len(self.cubes)):
            left_arm_start, right_arm_start = self.plan_single_cube_passing(i, self.cubes, left_arm_start, right_arm_start,env, bb, rrt_star_planner, transform_left_arm, transform_right_arm)


        t2 = time.time()
        print(f"It took t={t2 - start_time} seconds")
        # save the experiment to data:
        # Serializing json
        json_object = json.dumps(self.experiment_result, indent=4)
        # Writing to sample.json
        dir_path = r"./outputs/"
        with open(dir_path + "plan.json", "w") as outfile:
            outfile.write(json_object)
        # show the experiment then export it to a GIF
        visualizer.show_all_experiment(dir_path + "plan.json")
        visualizer.animate_by_pngs()

    def get_cubes_for_experiment(self, experiment_id, env):
        """
        Generates a list of initial cube positions for a specific experiment scenario.

        This method defines a 0.4m x 0.4m workspace grid and places cubes at specific
        coordinates based on the provided experiment ID. The coordinates are in world frame.

        Args:
            experiment_id (int): The identifier for the experiment scenario.
                - 1: A single cube scenario.
                - 2: A two-cube scenario.
            env (Environment): The environment object containing base offsets.

        Returns:
            list: A list of lists, where each inner list contains the [x, y, z] 
                  coordinates of a cube. The z-coordinate is set to half the 
                  cube's side length (0.02m) to place it on the surface.
        """
        cube_side = 0.04
        cubes = []
        offset = env.cube_area_corner[LocationType.RIGHT]
        if experiment_id == 1:
            x_min = 0.0
            x_max = 0.4
            y_min = 0.0
            y_max = 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0
            # row 1: cube 1
            pos = np.array([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
            cubes.append((pos + offset).tolist())
        elif experiment_id == 2:
            x_min = 0.0
            x_max = 0.4
            y_min = 0.0
            y_max = 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0
            # row 1: cube 1
            pos1 = np.array([x_min + 0.5 * x_slice, y_min + 1.5 * y_slice, cube_side / 2.0])
            cubes.append((pos1 + offset).tolist())
            # row 1: cube 2
            pos2 = np.array([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
            cubes.append((pos2 + offset).tolist())
        return cubes
