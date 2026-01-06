"""
Test script for UR5 robot IK and planning with GRID.
"""

import time
import numpy as np
from grid_robot_api.robot.arm.isaac_arm import IsaacArm

import grace

# Initialize robot arm
arm = IsaacArm()

# Get robot config - GRACE handles all URDF resolution internally
robot_config = grace.get_robot_config("ur5")


def generate_pointcloud(n_points: int = 3000):
    """Generate a simple table obstacle."""
    points = np.random.uniform(
        low=[-0.1, -0.4, -0.2],
        high=[0.7, 0.4, -0.1],
        size=(n_points, 3)
    )
    return points.tolist()

# Grid zero offset for UR5 - adjust based on your robot's calibration
GRID_ZERO_OFFSET = np.array(
    [
        8.210240487471765e-09,
        -1.7120000123977661,
        1.7120000123977661,
        9.11880988496705e-08,
        1.2503584834178127e-08,
        1.5700000524520874,
        0.0,
        6.88336471155182e-11,
    ],
    dtype=float,
)

### Create a start position

start_pos = [ {"pos": [0.2, -0.6, 0.2], "rpy": [0, 3.14, 0]} ]

### Compute IK for start position

result = grace.compute_ik(
    start_pos[0]["pos"],
    start_pos[0]["rpy"],
    orientation_type="rpy",
    robot="ur5",
)

### Move to start position

q_grid_start = np.array(result.config, dtype=float)
offset = GRID_ZERO_OFFSET[: q_grid_start.shape[0]]
q_cmd_start = q_grid_start - offset
arm.setJointAngles(list(q_cmd_start))
time.sleep(1.0)

### Set up target poses to reach
targets = [
    {"pos": [0.5, 0.15, 0.1], "rpy": [0, 3.14, 0]},
]

### Use GRACE's high-level API - handles IK + motion planning

for target in targets:
    print(f"\n{'='*60}")
    print(f"Planning to target: {target}")
    print(f"{'='*60}")

    result = grace.compute_plan(
        goal_position=target["pos"],
        goal_orientation=target["rpy"],
        orientation_type="rpy",
        robot="ur5",
        pointcloud=generate_pointcloud(),  
        planner="aorrtc",
        verbose=True,
    )

    if not result.success:
        print(f"Planning failed: {result.error}")
        continue

    ### Execute the planned path
    if result.path is not None:
        print(f"\nExecuting path with {len(result.path)} waypoints...")
        for q_grace in result.path:
            q_grid = np.array(q_grace, dtype=float)
            offset = GRID_ZERO_OFFSET[: q_grid.shape[0]]
            q_cmd = q_grid - offset
            arm.setJointAngles(list(q_cmd))
            time.sleep(0.1)
    else:
        # IK-only result
        q_grid = np.array(result.goal_config, dtype=float)
        offset = GRID_ZERO_OFFSET[: q_grid.shape[0]]
        q_cmd = q_grid - offset
        arm.setJointAngles(list(q_cmd))

    ### Wait and report final position
    time.sleep(10.0)
    q_actual = arm.getJointAngles()[: robot_config.arm_dof]
    pos = arm.getPosition()
    ori = arm.getOrientation()

    ### Use GRACE FK to verify the planned configuration
    fk = grace.compute_fk(result.goal_config, "ur5")

    print(f"\nResults:")
    print(f"  target: {target}")
    print(f"  goal_config: {result.goal_config.tolist()}")
    print(f"  q_actual: {q_actual}")
    print(f"  grace_fk_pos: {fk['position'].tolist()}")
    print(f"  grace_fk_quat: {fk['quaternion'].tolist()}")
    print(f"  grid_pos: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
    print(f"  grid_quat: [{ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}, {ori.w:.3f}]")
