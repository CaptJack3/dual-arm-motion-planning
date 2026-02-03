import numpy as np
from math import pi
from itertools import product

from environment import Environment, LocationType
from kinematics import UR5e_PARAMS, UR5e_without_camera_PARAMS, Transform
from building_blocks_gpt import BuildingBlocks3D
import inverse_kinematics


# ----------------------------
# Utilities
# ----------------------------
def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_from_T(T):
    """Yaw from rotation matrix assuming ZYX convention."""
    R = T[:3, :3]
    return np.arctan2(R[1, 0], R[0, 0])


def fk_pos_yaw(q):
    """Forward kinematics via provided IK module FK."""
    T = inverse_kinematics.forward_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, q)
    p = np.array(T[:3, 3], dtype=float)
    yaw = float(yaw_from_T(T))
    return p, yaw


def update_environment(env, active_arm, static_arm_conf, cubes_positions):
    env.set_active_arm(active_arm)
    env.update_obstacles(cubes_positions, static_arm_conf)


def deg2rad_list(vals_deg):
    return [np.deg2rad(v) for v in vals_deg]


def make_angle_grid(step_deg, low_deg=-180, high_deg=180):
    """Uniform grid in radians. high is excluded like np.arange."""
    vals = np.arange(low_deg, high_deg, step_deg, dtype=float)
    return np.deg2rad(vals)


# ----------------------------
# Core search
# ----------------------------
def find_one_config_for_target(
    env,
    bb_right,
    active_arm,
    static_arm_conf,
    cubes,
    target_p,
    target_yaw,
    frozen_joint_ids,      # e.g. [4,5] for q5,q6 (0-based)
    frozen_joint_values,   # radians, same length
    free_joint_ids,        # remaining joints
    grid_vals_rad,         # list/array of values used for each free joint (same for all)
    pos_tol=0.03,          # meters
    yaw_tol_deg=15.0,      # degrees
    max_checked=500000     # safety cap
):
    yaw_tol = np.deg2rad(yaw_tol_deg)

    # make env obstacles consistent with "right arm active"
    update_environment(env, active_arm, static_arm_conf, cubes)

    checked = 0

    # Cartesian product across 4 joints (if free_joint_ids has length 4)
    for free_vals in product(grid_vals_rad, repeat=len(free_joint_ids)):
        q = np.zeros(6, dtype=float)

        # set frozen
        for jid, val in zip(frozen_joint_ids, frozen_joint_values):
            q[jid] = val

        # set free
        for k, jid in enumerate(free_joint_ids):
            q[jid] = free_vals[k]

        checked += 1
        if checked > max_checked:
            return False, None, checked

        # collision / validity
        if not bb_right.config_validity_checker(q):
            continue

        # FK
        try:
            p, yaw = fk_pos_yaw(q)
        except Exception:
            continue

        # task check: position + yaw only
        if np.linalg.norm(p - target_p) <= pos_tol and abs(wrap_to_pi(yaw - target_yaw)) <= yaw_tol:
            return True, q, checked

    return False, None, checked


def try_freeze_two_joints_for_both_targets(
    env,
    bb_right,
    left_home,
    cubes,
    meeting_p,
    meeting_yaw,
    approach_p,
    approach_yaw,
    frozen_joint_ids=(4, 5),          # default: q5,q6
    frozen_values_grid_deg=(-180, -120, -60, 0, 60, 120),
    free_step_deg=15,
    pos_tol=0.03,
    yaw_tol_deg=15.0,
    per_target_max_checked=500000
):
    frozen_joint_ids = list(frozen_joint_ids)
    free_joint_ids = [i for i in range(6) if i not in frozen_joint_ids]

    grid_vals_rad = make_angle_grid(step_deg=free_step_deg, low_deg=-180, high_deg=180)

    print("\n==============================")
    print("NO-IK FREEZE SEARCH (RIGHT ARM)")
    print("==============================")
    print(f"Frozen joints (0-based): {frozen_joint_ids}  -> (1-based: {[j+1 for j in frozen_joint_ids]})")
    print(f"Frozen values grid (deg): {list(frozen_values_grid_deg)}")
    print(f"Free joints (0-based): {free_joint_ids}     -> (1-based: {[j+1 for j in free_joint_ids]})")
    print(f"Free grid step: {free_step_deg} deg  |  grid values per joint: {len(grid_vals_rad)}")
    print(f"Position tol: {pos_tol} m | Yaw tol: {yaw_tol_deg} deg")
    print("==============================\n")

    best = None

    for v0_deg in frozen_values_grid_deg:
        for v1_deg in frozen_values_grid_deg:
            frozen_vals = [np.deg2rad(v0_deg), np.deg2rad(v1_deg)]

            # --- meeting search ---
            ok_m, q_m, c_m = find_one_config_for_target(
                env=env,
                bb_right=bb_right,
                active_arm=LocationType.RIGHT,
                static_arm_conf=left_home,
                cubes=cubes,
                target_p=meeting_p,
                target_yaw=meeting_yaw,
                frozen_joint_ids=frozen_joint_ids,
                frozen_joint_values=frozen_vals,
                free_joint_ids=free_joint_ids,
                grid_vals_rad=grid_vals_rad,
                pos_tol=pos_tol,
                yaw_tol_deg=yaw_tol_deg,
                max_checked=per_target_max_checked
            )

            if not ok_m:
                print(f"Freeze {v0_deg:>4.0f},{v1_deg:>4.0f} deg -> meeting: FAIL (checked {c_m})")
                continue

            # --- approach search ---
            ok_a, q_a, c_a = find_one_config_for_target(
                env=env,
                bb_right=bb_right,
                active_arm=LocationType.RIGHT,
                static_arm_conf=left_home,
                cubes=cubes,
                target_p=approach_p,
                target_yaw=approach_yaw,
                frozen_joint_ids=frozen_joint_ids,
                frozen_joint_values=frozen_vals,
                free_joint_ids=free_joint_ids,
                grid_vals_rad=grid_vals_rad,
                pos_tol=pos_tol,
                yaw_tol_deg=yaw_tol_deg,
                max_checked=per_target_max_checked
            )

            if not ok_a:
                print(f"Freeze {v0_deg:>4.0f},{v1_deg:>4.0f} deg -> meeting: OK (checked {c_m}), approach: FAIL (checked {c_a})")
                continue

            # success for both
            print(f"\n✅ SUCCESS: freeze (deg) = [{v0_deg}, {v1_deg}] for joints {[j+1 for j in frozen_joint_ids]}")
            print(f"Meeting q (deg):  {np.round(np.rad2deg(q_m), 1).tolist()}")
            print(f"Approach q (deg): {np.round(np.rad2deg(q_a), 1).tolist()}\n")

            best = (v0_deg, v1_deg, q_m, q_a)
            return best  # stop at first success (fast)

    print("\n❌ No freeze pair in this grid worked for BOTH meeting and approach.")
    print("Try: (1) bigger tolerances, (2) smaller step (10deg), (3) different frozen joints.")
    return None


# ----------------------------
# Main
# ----------------------------
def main():
    # --- Build environment and BOTH arm transforms (required!) ---
    ur_right = UR5e_PARAMS(inflation_factor=1.0)
    ur_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)

    env = Environment(ur_params=ur_right)

    T_right = Transform(ur_params=ur_right, ur_location=env.arm_base_location[LocationType.RIGHT])
    T_left = Transform(ur_params=ur_left, ur_location=env.arm_base_location[LocationType.LEFT])

    env.arm_transforms[LocationType.RIGHT] = T_right
    env.arm_transforms[LocationType.LEFT] = T_left

    # Building blocks for RIGHT arm collision checks
    bb_right = BuildingBlocks3D(transform=T_right, ur_params=ur_right, env=env, resolution=0.1)

    # Static LEFT arm configuration (home)
    left_home = np.deg2rad([0, -90, 0, -90, 0, 0])

    # --- CUBES ---
    # If you want to ignore cubes, keep empty. If you want exact scene, add real cube positions.
    cubes = []  # e.g. [[x,y,z], [x,y,z]]

    # ============================================================
    # PUT HERE YOUR EXACT TARGETS (RIGHT ARM ONLY)
    # You said: use (p + yaw=t_z) only.
    # So give: meeting_p, meeting_yaw and approach_p, approach_yaw
    # ============================================================

    # Example: replace with YOUR real meeting target for right arm
    meeting_p = np.array([1.3631845004, 1.3346584665, 0.50], dtype=float)
    meeting_yaw = 0.0  # radians

    # Example: replace with YOUR real approach target for right arm
    approach_p = np.array([1.25, 1.25, 0.62], dtype=float)
    approach_yaw = 0.0  # radians

    # ============================================================
    # Run the freeze search
    # Choose which TWO joints to freeze:
    #   joints are 1..6 in human terms, but we pass 0..5
    # Good starting candidates: (q5,q6) => (4,5) or (q4,q6) => (3,5)
    # ============================================================
    result = try_freeze_two_joints_for_both_targets(
        env=env,
        bb_right=bb_right,
        left_home=left_home,
        cubes=cubes,
        meeting_p=meeting_p,
        meeting_yaw=meeting_yaw,
        approach_p=approach_p,
        approach_yaw=approach_yaw,
        frozen_joint_ids=(4, 5),                # freeze q5,q6
        frozen_values_grid_deg=(-180, -120, -60, 0, 60, 120),
        free_step_deg=15,
        pos_tol=0.03,
        yaw_tol_deg=15.0,
        per_target_max_checked=400000
    )

    if result is None:
        print("\nTry changing frozen joints to (3,5) i.e. q4,q6 or loosen tolerances.")
    else:
        v0_deg, v1_deg, q_m, q_a = result
        print("Best found (first success) freeze values:")
        print(f"  freeze joints (deg): {v0_deg}, {v1_deg}")


if __name__ == "__main__":
    main()
