import json
import time
import os
import re
from enum import Enum
import numpy as np

from environment import Environment, LocationType
from kinematics import UR5e_PARAMS, UR5e_without_camera_PARAMS, Transform
from planners import RRT_STAR
from building_blocks_gpt import BuildingBlocks3D
from visualizer import Visualize_UR
import inverse_kinematics

# IMPORTANT: used to hard-reset the planner between segments
from RRTTree_mod import RRTTree


def log(msg):
    written_log = f"STEP: {msg}"
    print(written_log)
    dir_path = r"./outputs/"
    os.makedirs(dir_path, exist_ok=True)
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
        self.cubes = cubes

        # tunable params
        self.max_itr = 1000
        self.max_step_size = 0.5
        self.goal_bias = 0.05
        self.resolution = 0.1

        # start confs
        self.right_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        self.left_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])

        self.experiment_result = []

        # will be set in meeting IK
        self.right_arm_meeting_conf = None
        self.left_arm_meeting_conf = None

    def push_step_info_into_single_cube_passing_data(
        self, description, active_id, command, static_conf, path, cubes, gripper_pre, gripper_post,
        movel_end_conf=None
    ):
        """
        path:
          - for "move": list of joint configs (each list len=6)
          - for "movel": relative translation [dx,dy,dz] (len=3)

        movel_end_conf:
          - optional end joint conf (len=6) computed via IK for visualization/debug
        """
        # convert numpy arrays to lists
        if isinstance(static_conf, np.ndarray):
            static_conf = static_conf.tolist()
        path = [p.tolist() if isinstance(p, np.ndarray) else p for p in path]
        cubes = [c.tolist() if isinstance(c, np.ndarray) else c for c in cubes]

        self.experiment_result[-1]["description"].append(description)
        self.experiment_result[-1]["active_id"].append(int(active_id))
        self.experiment_result[-1]["command"].append(command)
        self.experiment_result[-1]["static"].append(static_conf)
        self.experiment_result[-1]["path"].append(path)
        self.experiment_result[-1]["cubes"].append(cubes)
        self.experiment_result[-1]["gripper_pre"].append(gripper_pre)
        self.experiment_result[-1]["gripper_post"].append(gripper_post)

        # Non-breaking extra info (ignored by your executor if it doesn’t read it)
        if "movel_end_conf" not in self.experiment_result[-1]:
            self.experiment_result[-1]["movel_end_conf"] = []
        self.experiment_result[-1]["movel_end_conf"].append(
            None if movel_end_conf is None else [float(x) for x in movel_end_conf]
        )

    def _reset_planner_for_segment(self, planner: RRT_STAR):
        planner.goal_prob = self.goal_bias
        planner.goal_id = None
        planner.stop_on_goal = False
        planner.success_list = []
        planner.cost_list = []
        planner.tree = RRTTree(planner.bb)

    # -------------------------
    # Snapshot helper
    # -------------------------
    def _safe(self, s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^a-z0-9_\-]+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s[:160]

    def _snap(self, visualizer: Visualize_UR, tag: str, conf_right, conf_left, cubes_snapshot):
        os.makedirs("./outputs", exist_ok=True)
        fname = f"./outputs/{self._safe(tag)}.png"

        # clear and redraw one static frame
        visualizer.ax.clear()

        gR = visualizer.transform_right_arm.conf2sphere_coords(np.array(conf_right, dtype=float))
        gL = visualizer.transform_left_arm.conf2sphere_coords(np.array(conf_left, dtype=float))

        visualizer.draw_spheres(gR)
        visualizer.draw_spheres(gL)
        visualizer.draw_square()
        visualizer.draw_cubes(cubes_snapshot)
        visualizer.draw_obstacles()
        visualizer.ax.set_title(tag)

        visualizer.show()
        visualizer.fig.savefig(fname, dpi=300)

    # -------------------------
    # MOVEL end-conf via IK (for snapshots / debug)
    # -------------------------
    def _movel_end_conf(
        self,
        bb: BuildingBlocks3D,
        arm_transform: Transform,
        start_conf: np.ndarray,
        rel_xyz: np.ndarray,
        rpy_keep: np.ndarray,
        cubes_for_env: list,
        env: Environment,
        active_arm: LocationType,
        static_conf: np.ndarray,
    ) -> np.ndarray:
        """
        Compute end joint conf for a moveL_relative([dx,dy,dz]) step.
        Assumption: rel_xyz is expressed in TOOL frame.

        If your rel is expressed in BASE/WORLD frame, replace:
            delta_world = R_world_tool @ rel_xyz
        with:
            delta_world = rel_xyz
        """
        # update env collision set for this IK validation
        update_environment(env, active_arm, static_conf, cubes_for_env)

        # build delta in WORLD frame using tool orientation
        R_world_tool = arm_transform.rpy_to_rotation_matrix(float(rpy_keep[0]), float(rpy_keep[1]), float(rpy_keep[2]))
        delta_world = R_world_tool @ rel_xyz  # <-- CHANGE HERE if rel_xyz is NOT in tool frame

        # current TCP world position:
        # We do NOT rely on FK->position extraction here; instead we use the fact that you already
        # define rpy_keep and you can track the current "approach" position.
        # So we’ll approximate current TCP world position by reading it from the current config FK:
        T_base_tool = inverse_kinematics.forward_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, start_conf)
        T_world_tool = arm_transform.base_transform @ T_base_tool
        p_world = T_world_tool[0:3, 3].astype(float)

        p_target_world = p_world + delta_world

        # Build target in base frame using Transform helper
        T_target_base = arm_transform.get_base_to_tool_transform(position=p_target_world.tolist(), rpy=rpy_keep.tolist())

        IK = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_target_base)
        valid = bb.validate_IK_solutions(IK, T_target_base)
        if len(valid) == 0:
            raise RuntimeError("MOVEL IK failed: no valid IK solutions for end pose.")

        end_conf = min(valid, key=lambda q: np.linalg.norm(q - start_conf))
        return np.array(end_conf, dtype=float)

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
        self._reset_planner_for_segment(planner)

        path, cost = planner.find_path(start_conf=start_conf, goal_conf=goal_conf)
        if path is None:
            raise RuntimeError(f"Planner failed: no path found for segment '{description}' (active={active_id}).")

        self.push_step_info_into_single_cube_passing_data(
            description=description,
            active_id=active_id,
            command=command,
            static_conf=static_arm_conf.tolist(),
            path=[p.tolist() for p in path],
            cubes=[list(c) for c in cubes_real],
            gripper_pre=gripper_pre,
            gripper_post=gripper_post,
        )
        return np.array(path[-1], dtype=float)  # return final conf

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
        visualizer,
    ):
        # add a new step entry
        self.experiment_result.append({
            "description": [],
            "active_id": [],
            "command": [],
            "static": [],
            "path": [],
            "cubes": [],
            "gripper_pre": [],
            "gripper_post": [],
            # optional list created in push_step... : movel_end_conf
        })

        # ================
        # Segment 1: RIGHT start -> cube approach (LEFT static)
        # ================
        description = "right_arm => [start -> cube pickup], left_arm static"
        log(msg=description)
        self._snap(visualizer, f"cube{cube_i:02d}_S1_before_right_start", right_arm_start, left_arm_start, cubes)

        update_environment(env, LocationType.RIGHT, left_arm_start, cubes)

        cube_pos = np.array(cubes[cube_i], dtype=float)
        above_offset = 0.12
        p_approach = cube_pos + np.array([0.0, 0.0, above_offset], dtype=float)
        rpy_pick = np.array([np.pi, 0.0, 0.0], dtype=float)

        T_pick = right_arm_transform.get_base_to_tool_transform(position=p_approach, rpy=rpy_pick)
        IK = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_pick)
        valid = bb_right.validate_IK_solutions(IK, T_pick)
        if len(valid) == 0:
            raise RuntimeError(f"No valid IK for cube approach (cube {cube_i}).")

        cube_approach = min(valid, key=lambda q: np.linalg.norm(q - right_arm_start))
        cube_approach = np.array(cube_approach, dtype=float)

        right_arm_after = self.plan_single_arm(
            planner_right, right_arm_start, cube_approach, description,
            LocationType.RIGHT, "move", left_arm_start, cubes,
            Gripper.OPEN, Gripper.STAY
        )

        self._snap(visualizer, f"cube{cube_i:02d}_S1_after_right_cube_approach", right_arm_after, left_arm_start, cubes)

        # ================
        # MOVEL down to grasp (compute end conf via IK so we can snapshot)
        # ================
        rel_down = np.array([0.0, 0.0, -0.14], dtype=float)
        right_grasp_conf = self._movel_end_conf(
            bb=bb_right,
            arm_transform=right_arm_transform,
            start_conf=right_arm_after,
            rel_xyz=rel_down,
            rpy_keep=rpy_pick,
            cubes_for_env=cubes,
            env=env,
            active_arm=LocationType.RIGHT,
            static_conf=left_arm_start,
        )

        self.push_step_info_into_single_cube_passing_data(
            description="picking up a cube: go down",
            active_id=LocationType.RIGHT,
            command="movel",
            static_conf=left_arm_start.tolist(),
            path=rel_down.tolist(),               # keep executor format
            cubes=[list(c) for c in cubes],
            gripper_pre=Gripper.STAY,
            gripper_post=Gripper.CLOSE,
            movel_end_conf=right_grasp_conf.tolist(),
        )
        self._snap(visualizer, f"cube{cube_i:02d}_S1_grasp_down_end", right_grasp_conf, left_arm_start, cubes)

        # MOVEL up after grasp
        rel_up = np.array([0.0, 0.0, 0.14], dtype=float)
        right_back_up_conf = self._movel_end_conf(
            bb=bb_right,
            arm_transform=right_arm_transform,
            start_conf=right_grasp_conf,
            rel_xyz=rel_up,
            rpy_keep=rpy_pick,
            cubes_for_env=cubes,
            env=env,
            active_arm=LocationType.RIGHT,
            static_conf=left_arm_start,
        )

        self.push_step_info_into_single_cube_passing_data(
            description="picking up a cube: go up",
            active_id=LocationType.RIGHT,
            command="movel",
            static_conf=left_arm_start.tolist(),
            path=rel_up.tolist(),
            cubes=[list(c) for c in cubes],
            gripper_pre=Gripper.STAY,
            gripper_post=Gripper.STAY,
            movel_end_conf=right_back_up_conf.tolist(),
        )
        self._snap(visualizer, f"cube{cube_i:02d}_S1_grasp_up_end", right_back_up_conf, left_arm_start, cubes)

        # Remove carried cube from obstacle list during transfer motion
        cubes_wo_i = [cubes[j] for j in range(len(cubes)) if j != cube_i]

        # ================
        # Segment 2: RIGHT -> meeting (LEFT static)
        # ================
        description = "right_arm => [cube pickup -> meeting], left_arm static"
        log(msg=description)
        self._snap(visualizer, f"cube{cube_i:02d}_S2_before_right_to_meeting", right_back_up_conf, left_arm_start, cubes_wo_i)

        right_at_meeting = self.plan_single_arm(
            planner_right, right_back_up_conf, np.array(self.right_arm_meeting_conf, dtype=float),
            description, LocationType.RIGHT, "move", left_arm_start, cubes_wo_i,
            Gripper.STAY, Gripper.STAY
        )

        self._snap(visualizer, f"cube{cube_i:02d}_S2_after_right_at_meeting", right_at_meeting, left_arm_start, cubes_wo_i)

        # ================
        # Segment 3: LEFT -> meeting (RIGHT static)
        # ================
        description = "left_arm => [start -> meeting], right_arm static"
        log(msg=description)
        self._snap(visualizer, f"cube{cube_i:02d}_S3_before_left_to_meeting", right_at_meeting, left_arm_start, cubes_wo_i)

        left_at_meeting = self.plan_single_arm(
            planner_left, left_arm_start, np.array(self.left_arm_meeting_conf, dtype=float),
            description, LocationType.LEFT, "move", right_at_meeting, cubes_wo_i,
            Gripper.STAY, Gripper.OPEN
        )

        self._snap(visualizer, f"cube{cube_i:02d}_S3_after_left_at_meeting", right_at_meeting, left_at_meeting, cubes_wo_i)

        # ================
        # Segment 4: handover micro movels (compute IK end confs for snapshots)
        # ================
        # Here rpy_keep should be the meeting orientation you used in meeting IK (use same rpys from plan_experiment)
        # We don’t have them stored, so we approximate using current FK orientation; if you want exact, store rpy in self.
        # For now: use the FK orientation extracted into rpy is hard, so we keep the meeting orientation constant as:
        #   right: rpy_right_meet , left: rpy_left_meet
        # If you want, I can store these in self.* during meeting computation.
        rpy_right_meet = np.array([0.0, 0.0, 0.0], dtype=float)  # placeholder (snapshots still OK-ish for small deltas)
        rpy_left_meet = np.array([0.0, 0.0, 0.0], dtype=float)

        cubes_wo_i_ll = [list(c) for c in cubes_wo_i]

        # right approaches
        rel = np.array([-0.06, 0.0, 0.0], dtype=float)
        right_approach_conf = self._movel_end_conf(
            bb_right, right_arm_transform, right_at_meeting, rel, rpy_right_meet,
            cubes_wo_i, env, LocationType.RIGHT, left_at_meeting
        )
        self.push_step_info_into_single_cube_passing_data(
            "handover: right arm approaches",
            LocationType.RIGHT,
            "movel",
            left_at_meeting.tolist(),
            rel.tolist(),
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.STAY,
            movel_end_conf=right_approach_conf.tolist(),
        )
        self._snap(visualizer, f"cube{cube_i:02d}_S4_right_approach_end", right_approach_conf, left_at_meeting, cubes_wo_i)

        # left approaches + grasps
        rel = np.array([-0.06, 0.0, 0.0], dtype=float)
        left_approach_conf = self._movel_end_conf(
            bb_left, left_arm_transform, left_at_meeting, rel, rpy_left_meet,
            cubes_wo_i, env, LocationType.LEFT, right_approach_conf
        )
        self.push_step_info_into_single_cube_passing_data(
            "handover: left arm approaches + grasps",
            LocationType.LEFT,
            "movel",
            right_approach_conf.tolist(),
            rel.tolist(),
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.CLOSE,
            movel_end_conf=left_approach_conf.tolist(),
        )
        self._snap(visualizer, f"cube{cube_i:02d}_S4_left_grasp_end", right_approach_conf, left_approach_conf, cubes_wo_i)

        # right releases + retreats
        rel = np.array([0.06, 0.0, 0.0], dtype=float)
        right_retreat_conf = self._movel_end_conf(
            bb_right, right_arm_transform, right_approach_conf, rel, rpy_right_meet,
            cubes_wo_i, env, LocationType.RIGHT, left_approach_conf
        )
        self.push_step_info_into_single_cube_passing_data(
            "handover: right releases + retreats",
            LocationType.RIGHT,
            "movel",
            left_approach_conf.tolist(),
            rel.tolist(),
            cubes_wo_i_ll,
            Gripper.OPEN,
            Gripper.STAY,
            movel_end_conf=right_retreat_conf.tolist(),
        )
        self._snap(visualizer, f"cube{cube_i:02d}_S4_right_retreat_end", right_retreat_conf, left_approach_conf, cubes_wo_i)

        # left retreats
        rel = np.array([0.06, 0.0, 0.0], dtype=float)
        left_retreat_conf = self._movel_end_conf(
            bb_left, left_arm_transform, left_approach_conf, rel, rpy_left_meet,
            cubes_wo_i, env, LocationType.LEFT, right_retreat_conf
        )
        self.push_step_info_into_single_cube_passing_data(
            "handover: left retreats",
            LocationType.LEFT,
            "movel",
            right_retreat_conf.tolist(),
            rel.tolist(),
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.STAY,
            movel_end_conf=left_retreat_conf.tolist(),
        )
        self._snap(visualizer, f"cube{cube_i:02d}_S4_left_retreat_end", right_retreat_conf, left_retreat_conf, cubes_wo_i)

        # ================
        # Segment 5: LEFT meeting -> place approach (IK + RRT*)
        # ================
        right_corner = np.array(env.cube_area_corner[LocationType.RIGHT], dtype=float)
        left_corner = np.array(env.cube_area_corner[LocationType.LEFT], dtype=float)

        cube_pos = np.array(cubes[cube_i], dtype=float)
        local_offset = cube_pos - right_corner
        place_cube_pos = left_corner + local_offset

        cube_side = 0.04
        place_cube_pos[2] = cube_side / 2.0

        above_place = 0.25
        p_place_approach = place_cube_pos + np.array([0.0, 0.0, above_place], dtype=float)
        rpy_place = np.array([np.pi, 0.0, 0.0], dtype=float)

        update_environment(env, LocationType.LEFT, right_retreat_conf, cubes_wo_i)
        T_place = left_arm_transform.get_base_to_tool_transform(position=p_place_approach, rpy=rpy_place)
        IKp = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_place)

        validp = bb_left.validate_IK_solutions(IKp, T_place)
        if len(validp) == 0:
            raise RuntimeError("No valid IK for left place-approach.")

        left_place_approach = min(validp, key=lambda q: np.linalg.norm(q - left_retreat_conf))
        left_place_approach = np.array(left_place_approach, dtype=float)

        description = "left_arm => [meeting -> place approach], right_arm static"
        log(msg=description)
        self._snap(visualizer, f"cube{cube_i:02d}_S5_before_left_to_place", right_retreat_conf, left_retreat_conf, cubes_wo_i)

        left_at_place = self.plan_single_arm(
            planner_left, left_retreat_conf, left_place_approach,
            description, LocationType.LEFT, "move", right_retreat_conf, cubes_wo_i,
            Gripper.STAY, Gripper.STAY
        )

        self._snap(visualizer, f"cube{cube_i:02d}_S5_after_left_place_approach", right_retreat_conf, left_at_place, cubes_wo_i)

        # MOVEL down + open (place)
        rel_down = np.array([0.0, 0.0, -0.25], dtype=float)
        left_drop_conf = self._movel_end_conf(
            bb_left, left_arm_transform, left_at_place, rel_down, rpy_place,
            cubes_wo_i, env, LocationType.LEFT, right_retreat_conf
        )
        self.push_step_info_into_single_cube_passing_data(
            "dropping a cube: go down",
            LocationType.LEFT,
            "movel",
            right_retreat_conf.tolist(),
            rel_down.tolist(),
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.OPEN,
            movel_end_conf=left_drop_conf.tolist(),
        )
        self._snap(visualizer, f"cube{cube_i:02d}_S5_drop_down_end", right_retreat_conf, left_drop_conf, cubes_wo_i)

        # MOVEL up
        rel_up = np.array([0.0, 0.0, 0.25], dtype=float)
        left_up_conf = self._movel_end_conf(
            bb_left, left_arm_transform, left_drop_conf, rel_up, rpy_place,
            cubes_wo_i, env, LocationType.LEFT, right_retreat_conf
        )
        self.push_step_info_into_single_cube_passing_data(
            "dropping a cube: go up",
            LocationType.LEFT,
            "movel",
            right_retreat_conf.tolist(),
            rel_up.tolist(),
            cubes_wo_i_ll,
            Gripper.STAY,
            Gripper.STAY,
            movel_end_conf=left_up_conf.tolist(),
        )
        self._snap(visualizer, f"cube{cube_i:02d}_S5_drop_up_end", right_retreat_conf, left_up_conf, cubes_wo_i)

        # Update cube position for next iteration
        cubes[cube_i] = place_cube_pos.tolist()

        # return ends (use joint confs)
        left_arm_end = np.array(left_up_conf, dtype=float)
        right_arm_end = np.array(right_retreat_conf, dtype=float)
        return left_arm_end, right_arm_end

    def plan_experiment(self):
        start_time = time.time()

        exp_id = 2
        ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
        ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)

        env = Environment(ur_params=ur_params_right)

        right_arm_rot = [0, 0, -np.pi / 2]
        left_arm_rot = [0, 0, np.pi / 2]

        transform_right_arm = Transform(
            ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT], ur_rotation=right_arm_rot
        )
        transform_left_arm = Transform(
            ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT], ur_rotation=left_arm_rot
        )

        env.arm_transforms[LocationType.RIGHT] = transform_right_arm
        env.arm_transforms[LocationType.LEFT] = transform_left_arm

        bb_right = BuildingBlocks3D(transform=transform_right_arm, ur_params=ur_params_right, env=env, resolution=self.resolution)
        bb_left = BuildingBlocks3D(transform=transform_left_arm, ur_params=ur_params_left, env=env, resolution=self.resolution)

        planner_right = RRT_STAR(max_step_size=self.max_step_size, max_itr=self.max_itr, bb=bb_right)
        planner_left = RRT_STAR(max_step_size=self.max_step_size, max_itr=self.max_itr, bb=bb_left)

        visualizer = Visualize_UR(
            ur_params_right,
            env=env,
            transform_right_arm=transform_right_arm,
            transform_left_arm=transform_left_arm,
        )

        if self.cubes is None:
            self.cubes = self.get_cubes_for_experiment(exp_id, env)

        log(msg="calculate meeting point for the test.")

        # Meeting point deterministic
        bL = np.array(env.arm_base_location[LocationType.LEFT], dtype=float)
        bR = np.array(env.arm_base_location[LocationType.RIGHT], dtype=float)

        M = np.array([(bL[0] + bR[0]) / 2.0, (bL[1] + bR[1]) / 2.0, 0.50], dtype=float)
        u = (bL - bR)
        u = u / np.linalg.norm(u)

        delta = 0.05

        u = np.array([0,1,0]) # Change only in y direction

        pR = M - delta * u
        pL = M + delta * u

        # (your meeting rpys)
        rpy_right = [-np.pi / 2, 0.0, 0.0]
        rpy_left = [np.pi / 2, np.pi / 2, 0.0]

        # RIGHT meeting IK
        update_environment(env, LocationType.RIGHT, self.left_arm_home, self.cubes)
        T_R = transform_right_arm.get_base_to_tool_transform(position=pR, rpy=rpy_right)
        IK_R = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_R)
        valid_R = bb_right.validate_IK_solutions(IK_R, T_R)
        if len(valid_R) == 0:
            raise RuntimeError("No valid RIGHT meeting IK solutions.")
        self.right_arm_meeting_conf = min(valid_R, key=lambda q: np.linalg.norm(q - self.right_arm_home))

        # LEFT meeting IK
        update_environment(env, LocationType.LEFT, self.right_arm_home, self.cubes)
        T_L = transform_left_arm.get_base_to_tool_transform(position=pL, rpy=rpy_left)
        IK_L = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, T_L)
        valid_L = bb_left.validate_IK_solutions(IK_L, T_L)
        if len(valid_L) == 0:
            raise RuntimeError("No valid LEFT meeting IK solutions.")
        self.left_arm_meeting_conf = min(valid_L, key=lambda q: np.linalg.norm(q - self.left_arm_home))

        log(msg=f"Meeting M = {M.tolist()}, delta={delta}")
        log(msg=f"Right meeting target pR = {pR.tolist()}, rpy_right={rpy_right}")
        log(msg=f"Left  meeting target pL = {pL.tolist()}, rpy_left={rpy_left}")

        # Run experiment
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
                visualizer,
            )

        t2 = time.time()
        print(f"It took t={t2 - start_time} seconds")

        # save experiment json
        '''
        os.makedirs("./outputs", exist_ok=True)
        dir_path = r"./outputs/"
        json_object = json.dumps(self.experiment_result, indent=4)
        with open(dir_path + "plan.json", "w") as outfile:
            outfile.write(json_object)

        # Keep your animation if you want
        visualizer.show_all_experiment(dir_path + "plan.json")
        visualizer.animate_by_pngs()
        '''




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

#
# if __name__ == "__main__":
#     exp = Experiment()
#     exp.plan_experiment()
