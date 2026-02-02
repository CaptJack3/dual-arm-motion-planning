import json
import time
from enum import Enum
import numpy as np
from matplotlib.sankey import RIGHT  # (kept as you had it; not required)

from environment import Environment, LocationType
from kinematics import UR5e_PARAMS, UR5e_without_camera_PARAMS, Transform
from planners import RRT_STAR
from building_blocks_gpt import BuildingBlocks3D
from visualizer import Visualize_UR
import inverse_kinematics

# IMPORTANT: used to hard-reset the planner between segments (recommended with your current planners.py)
from RRTTree_mod import RRTTree


def log(msg):
    written_log = f"STEP: {msg}"
    print(written_log)
    dir_path = r"./outputs/"
    with open(dir_path + "output.txt", "a") as file:
        file.write(f"{written_log}\n")


class Gripper(str, Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    STAY = "STAY"


def update_environment(env, active_arm, static_arm_conf, cubes_positions):
    env.set_active_arm(active_arm)
    env.update_obstacles(cubes_positions, static_arm_conf)


class Experiment:
    def __init__(self, cubes=None):
        # environment params
        self.cubes = cubes
        self.right_arm_base_delta = 0
        self.left_arm_base_delta = 0
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

        # will be set in TODO1
        self.right_arm_meeting_conf = None
        self.left_arm_meeting_conf = None

    def push_step_info_into_single_cube_passing_data(
        self, description, active_id, command, static_conf, path, cubes, gripper_pre, gripper_post
    ):
        # convert numpy arrays to lists
        if isinstance(static_conf, np.ndarray):
            static_conf = static_conf.tolist()
        path = [p.tolist() if isinstance(p, np.ndarray) else p for p in path]
        cubes = [c.tolist() if isinstance(c, np.ndarray) else c for c in cubes]

        self.experiment_result[-1]["description"].append(description)
        self.experiment_result[-1]["active_id"].append(active_id)
        self.experiment_result[-1]["command"].append(command)
        self.experiment_result[-1]["static"].append(static_conf)
        self.experiment_result[-1]["path"].append(path)
        self.experiment_result[-1]["cubes"].append(cubes)
        self.experiment_result[-1]["gripper_pre"].append(gripper_pre)
        self.experiment_result[-1]["gripper_post"].append(gripper_post)

    def _reset_planner_for_segment(self, planner: RRT_STAR):
        """
        Your current planners.py:
        - doesn't initialize goal_prob / goal_id / lists
        - and the tree keeps state unless reset
        So we hard-reset per segment to keep behavior correct and reproducible.
        """
        planner.goal_prob = self.goal_bias
        planner.goal_id = None
        planner.stop_on_goal = False
        planner.success_list = []
        planner.cost_list = []
        planner.tree = RRTTree(planner.bb)

    def plan_single_arm(
        self,
        planner,
        start_conf,
        goal_conf,
        description,
        active_id,
        command,
        static_arm_conf,
        cubes_real,
        gripper_pre,
        gripper_post,
    ):
        # Reset planner per segment (prevents cross-contamination between calls)
        self._reset_planner_for_segment(planner)

        # Your planners.py signature is find_path(self, start_conf, goal_conf) (NO manipulator arg)
        path,cost = planner.find_path(start_conf=start_conf, goal_conf=goal_conf)

        if path is None:
            raise RuntimeError(f"Planner failed: no path found for segment '{description}' (active={active_id}).")

        # create the arm plan
        self.push_step_info_into_single_cube_passing_data(
            description,
            active_id,
            command,
            static_arm_conf.tolist(),
            [path_element.tolist() for path_element in path],
            [list(cube) for cube in cubes_real],
            gripper_pre,
            gripper_post,
        )

    def plan_single_cube_passing(
        self,
        cube_i,
        cubes,
        left_arm_start,
        right_arm_start,
        env,
        bb_left,
        bb_right,
        planner_left,
        planner_right,
        left_arm_transform,
        right_arm_transform,
    ):
        # add a new step entry
        single_cube_passing_info = {
            "description": [],
            "active_id": [],
            "command": [],
            "static": [],
            "path": [],
            "cubes": [],
            "gripper_pre": [],
            "gripper_post": [],
        }
        self.experiment_result.append(single_cube_passing_info)

        # ============================
        # Segment 1: RIGHT start -> cube pickup approach (LEFT static)
        # ============================
        description = "right_arm => [start -> cube pickup], left_arm static"
        active_arm = LocationType.RIGHT
        log(msg=description)

        update_environment(env, LocationType.RIGHT, left_arm_start, cubes)

        cube_pos = np.array(cubes[cube_i], dtype=float)
        above_offset = 0.12
        p_approach = cube_pos + np.array([0.0, 0.0, above_offset], dtype=float)
        rpy_pick = [np.pi, 0.0, 0.0]

        T_pick = right_arm_transform.get_base_to_tool_transform(position=p_approach, rpy=rpy_pick)
        IK = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_pick)

        valid = bb_right.validate_IK_solutions(IK, T_pick)
        if len(valid) == 0:
            raise RuntimeError(
                f"TODO2 failed: no valid IK for cube approach (cube {cube_i}). "
                "Try changing above_offset or rpy_pick."
            )

        cube_approach = min(valid, key=lambda q: np.linalg.norm(q - right_arm_start))

        self.plan_single_arm(
            planner_right,
            right_arm_start,
            cube_approach,
            description,
            active_arm,
            "move",
            left_arm_start,
            cubes,
            Gripper.OPEN,
            Gripper.STAY,
        )

        # down to grasp

        '''
                self, description, active_id, command, static_conf, path, cubes, gripper_pre, gripper_post
    ):
        self.experiment_result[-1]["description"].append(description)
        self.experiment_result[-1]["active_id"].append(active_id)
        self.experiment_result[-1]["command"].append(command)
        self.experiment_result[-1]["static"].append(static_conf)
        self.experiment_result[-1]["path"].append(path)
        self.experiment_result[-1]["cubes"].append(cubes)
        self.experiment_result[-1]["gripper_pre"].append(gripper_pre)
        self.experiment_result[-1]["gripper_post"].append(gripper_post)

        
        
        '''
        self.push_step_info_into_single_cube_passing_data(
            "picking up a cube: go down",
            LocationType.RIGHT,
            "movel",
            left_arm_start. tolist(),
            # cube_approach, # [0, 0, -0.14],
            [0, 0, -0.14],
            [list(cube) for cube in cubes],
            Gripper.STAY,
            Gripper.CLOSE,
        )

        # up after grasp
        cubes_ll = [list(cube) for cube in cubes]
        self.push_step_info_into_single_cube_passing_data(
            "picking up a cube: go up",
            LocationType.RIGHT,
            "movel",
            left_arm_start.tolist(),
            [0, 0, 0.14],
            cubes_ll,
            Gripper.STAY,
            Gripper.STAY,
        )

        # Remove carried cube from obstacle list during transfer motion
        cubes_wo_i = [cubes[j] for j in range(len(cubes)) if j != cube_i]
        cubes_wo_i_ll = [list(cube) for cube in cubes_wo_i]

        # ============================
        # Segment 2: RIGHT cube_approach -> RIGHT meeting (LEFT static)
        # ============================
        description = "right_arm => [cube pickup -> meeting], left_arm static"
        log(msg=description)
        update_environment(env, LocationType.RIGHT, left_arm_start, cubes_wo_i)

        self.plan_single_arm(
            planner_right,
            cube_approach,
            self.right_arm_meeting_conf,
            description,
            LocationType.RIGHT,
            "move",
            left_arm_start,
            cubes_wo_i,
            Gripper.STAY,
            Gripper.STAY,
        )
        right_at_meeting = np.array(self.right_arm_meeting_conf, dtype=float)

        # ============================
        # Segment 3: LEFT start -> LEFT meeting (RIGHT static)
        # ============================
        description = "left_arm => [start -> meeting], right_arm static"
        log(msg=description)
        update_environment(env, LocationType.LEFT, right_at_meeting, cubes_wo_i)

        self.plan_single_arm(
            planner_left,
            left_arm_start,
            self.left_arm_meeting_conf,
            description,
            LocationType.LEFT,
            "move",
            right_at_meeting,
            cubes_wo_i,
            Gripper.STAY,
            Gripper.OPEN,
        )
        left_at_meeting = np.array(self.left_arm_meeting_conf, dtype=float)

        # ============================
        # Segment 4: handover micro motions (movel)
        # ============================
        self.push_step_info_into_single_cube_passing_data(
            "handover: right arm approaches",
            LocationType.RIGHT,
            "movel",
            left_at_meeting.tolist(),
            [-0.06, 0, 0],
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.STAY,
        )
        self.push_step_info_into_single_cube_passing_data(
            "handover: left arm approaches + grasps",
            LocationType.LEFT,
            "movel",
            right_at_meeting.tolist(),
            [0.06, 0, 0], #[-0.06, 0, 0]?
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.CLOSE,
        )
        self.push_step_info_into_single_cube_passing_data(
            "handover: right releases + retreats",
            LocationType.RIGHT,
            "movel",
            left_at_meeting.tolist(),
            [0.06, 0, 0],
            cubes_wo_i_ll,
            Gripper.OPEN,
            Gripper.STAY,
        )
        self.push_step_info_into_single_cube_passing_data(
            "handover: left retreats",
            LocationType.LEFT,
            "movel",
            right_at_meeting.tolist(),
            [-0.06, 0, 0], # [0.06, 0, 0]?
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.STAY,
        )

        # ============================
        # Segment 5: LEFT meeting -> place approach (IK + RRT*)
        # ============================
        right_corner = np.array(env.cube_area_corner[LocationType.RIGHT], dtype=float)
        left_corner = np.array(env.cube_area_corner[LocationType.LEFT], dtype=float)

        cube_pos = np.array(cubes[cube_i], dtype=float)
        local_offset = cube_pos - right_corner
        place_cube_pos = left_corner + local_offset

        cube_side = 0.04
        place_cube_pos[2] = cube_side / 2.0

        above_place = 0.25
        p_place_approach = place_cube_pos + np.array([0.0, 0.0, above_place], dtype=float)
        rpy_place = [np.pi, 0.0, 0.0]
        rpy_left = [np.pi, np.pi / 2, np.pi]  # TODO - not sure if right


        update_environment(env, LocationType.LEFT, right_at_meeting, cubes_wo_i)
        T_place = left_arm_transform.get_base_to_tool_transform(position=p_place_approach, rpy=rpy_place)
        IKp = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_place)

        validp = bb_left.validate_IK_solutions(IKp, T_place)
        if len(validp) == 0:
            raise RuntimeError("TODO3: no valid IK for cleft place-approach. Try changing above_place or rpy_place.")

        left_place_approach = min(validp, key=lambda q: np.linalg.norm(q - left_at_meeting))

        description = "left_arm => [meeting -> place approach], right_arm static"
        log(msg=description)
        update_environment(env, LocationType.LEFT, right_at_meeting, cubes_wo_i)

        self.plan_single_arm(
            planner_left,
            left_at_meeting,
            left_place_approach,
            description,
            LocationType.LEFT,
            "move",
            right_at_meeting,
            cubes_wo_i,
            Gripper.STAY,
            Gripper.STAY,
        )

        # down + open, then up
        self.push_step_info_into_single_cube_passing_data(
            "dropping a cube: go down",
            LocationType.LEFT,
            "movel",
            right_at_meeting.tolist(),
            [0, 0, -0.25],
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.OPEN,
        )
        self.push_step_info_into_single_cube_passing_data(
            "dropping a cube: go up",
            LocationType.LEFT,
            "movel",
            right_at_meeting.tolist(),
            [0, 0, 0.25],
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.STAY,
        )

        # Update cube position for next iteration
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
        right_arm_rot =[0 ,0 , -np.pi/2]
        left_arm_rot =[0 ,0 , np.pi/2]


        transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT],ur_rotation=right_arm_rot)
        transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT],ur_rotation=left_arm_rot)

        env.arm_transforms[LocationType.RIGHT] = transform_right_arm
        env.arm_transforms[LocationType.LEFT] = transform_left_arm

        # Building blocks: one per arm
        bb_right = BuildingBlocks3D(
            transform=transform_right_arm,
            ur_params=ur_params_right,
            env=env,
            resolution=self.resolution,
        )
        bb_left = BuildingBlocks3D(
            transform=transform_left_arm,
            ur_params=ur_params_left,
            env=env,
            resolution=self.resolution,
        )

        # Planners: one per arm
        planner_right = RRT_STAR(
            max_step_size=self.max_step_size,
            max_itr=self.max_itr,
            bb=bb_right,
        )
        planner_left = RRT_STAR(
            max_step_size=self.max_step_size,
            max_itr=self.max_itr,
            bb=bb_left,
        )

        visualizer = Visualize_UR(
            ur_params_right,
            env=env,
            transform_right_arm=transform_right_arm,
            transform_left_arm=transform_left_arm,
        )

        # cubes
        if self.cubes is None:
            self.cubes = self.get_cubes_for_experiment(exp_id, env)

        log(msg="calculate meeting point for the test.")

        # ============================
        # TODO 1: meeting confs (deterministic)
        # ============================
        bL = np.array(env.arm_base_location[LocationType.LEFT], dtype=float)
        bR = np.array(env.arm_base_location[LocationType.RIGHT], dtype=float)

        M = np.array([(bL[0] + bR[0]) / 2.0, (bL[1] + bR[1]) / 2.0, 0.50], dtype=float)

        u = (bL - bR)
        u = u / np.linalg.norm(u)
        u=np.array([0.0,1.0,0.0])
        delta = 0.1
        delta_vec = delta * u

        pR = M - delta_vec
        pL = M + delta_vec

        rpy_right = [-np.pi/2, 0.0, 0.0]
        # rpy_right = [0.0, 0.0, np.pi/2] #YOTAM
        # rpy_left = [rpy_right[0], rpy_right[1], rpy_right[2] + np.pi / 2] # TODO - that is the right notation of rpy_right

        # same down orientation, rotated around Z to face different side
        # rpy_left = [np.pi, 0.0, np.pi / 2]  # 90° around Z

        # also tilt around Y (changes face / approach)
        rpy_left = [-np.pi/2, np.pi , np.pi]  # your requested type of change
        rpy_left = [np.pi/2,np.pi/2,0.0]  # YOTAM your requested type of change THIS ONE SEEMS TO WORK
        # rpy_left = [-np.pi/2,0.0,np.pi/2]  # YOTAM your requested type of change CURRNENTLY USED

        # ---- RIGHT ARM IK ----
        update_environment(env, LocationType.RIGHT, self.left_arm_home, self.cubes)
        T_R = transform_right_arm.get_base_to_tool_transform(position=pR, rpy=rpy_right)
        IK_R = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_R)
        valid_R = bb_right.validate_IK_solutions(IK_R, T_R)
        if len(valid_R) == 0:
            raise RuntimeError("TODO1: No valid RIGHT meeting IK solutions. Try changing rpy_right.")

        self.right_arm_meeting_conf = min(valid_R, key=lambda q: np.linalg.norm(q - self.right_arm_home))

        # ---- LEFT ARM IK ----
        update_environment(env, LocationType.LEFT, self.right_arm_home, self.cubes)
        T_L = transform_left_arm.get_base_to_tool_transform(position=pL, rpy=rpy_left)
        IK_L = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_L)
        valid_L = bb_left.validate_IK_solutions(IK_L, T_L)
        if len(valid_L) == 0:
            raise RuntimeError("TODO1: No valid LEFT meeting IK solutions. Try changing rpy_left.")

        self.left_arm_meeting_conf = min(valid_L, key=lambda q: np.linalg.norm(q - self.left_arm_home))

        log(msg=f"Meeting M = {M.tolist()}, delta={delta}")
        log(msg=f"Right meeting target pR = {pR.tolist()}, rpy_right={rpy_right}")
        log(msg=f"Left  meeting target pL = {pL.tolist()}, rpy_left={rpy_left}")

        # ============================
        # Run experiment
        # ============================
        log(msg="start planning the experiment.")
        left_arm_start = self.left_arm_home
        right_arm_start = self.right_arm_home

        for i in range(len(self.cubes)):
            left_arm_start, right_arm_start = self.plan_single_cube_passing(
                i,
                self.cubes,
                left_arm_start,
                right_arm_start,
                env,
                bb_left,
                bb_right,
                planner_left,
                planner_right,
                transform_left_arm,
                transform_right_arm,
            )

        t2 = time.time()
        print(f"It took t={t2 - start_time} seconds")

        # save the experiment to data:
        json_object = json.dumps(self.experiment_result, indent=4)
        dir_path = r"./outputs/"
        with open(dir_path + "plan.json", "w") as outfile:
            outfile.write(json_object)

        # show the experiment then export it to a GIF
        visualizer.show_all_experiment(dir_path + "plan.json")
        visualizer.animate_by_pngs()

    def get_cubes_for_experiment(self, experiment_id, env):
        cube_side = 0.04
        cubes = []
        offset = env.cube_area_corner[LocationType.RIGHT]

        if experiment_id == 1:
            x_min, x_max = 0.0, 0.4
            y_min, y_max = 0.0, 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0
            pos = np.array([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
            cubes.append((pos + offset).tolist())

        elif experiment_id == 2:
            x_min, x_max = 0.0, 0.4
            y_min, y_max = 0.0, 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0

            pos1 = np.array([x_min + 0.5 * x_slice, y_min + 1.5 * y_slice, cube_side / 2.0])
            cubes.append((pos1 + offset).tolist())

            pos2 = np.array([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
            cubes.append((pos2 + offset).tolist())

        return cubes
