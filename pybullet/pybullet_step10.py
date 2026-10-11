import pybullet as p
import pybullet_data
import time

print("--- 🏆 STEP 10: THE SMART TRASH COLLECTOR (Finale) ---")

# 1. SETUP THE WORLD
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# 2. SPAWN OUR ACTORS
# The Franka Panda robot
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

# The "Trash" (a small red block)
trash_start_pos = [0.5, 0, 0.05]
trash_id = p.loadURDF("cube_small.urdf", trash_start_pos)
p.changeVisualShape(trash_id, -1, rgbaColor=[1, 0, 0, 1]) # Make it red!

# The "Bin" (we'll use a larger tray/box as the goal)
bin_pos = [0, 0.5, 0.05]
# Let's create a simple visual box to represent the bin
bin_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.05], rgbaColor=[0.2, 0.2, 0.2, 1])
bin_id = p.createMultiBody(baseVisualShapeIndex=bin_visual, basePosition=bin_pos)

# Robot indices for Panda
ee_index = 11
finger_1 = 9
finger_2 = 10

def control_robot(target_ee_pos, finger_pos, duration_seconds):
    """Helper function to move the robot's arm and fingers over time."""
    steps = int(duration_seconds * 240)
    for _ in range(steps):
        # IK for the arm
        joint_angles = p.calculateInverseKinematics(robot_id, ee_index, target_ee_pos)
        for i in range(9):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_angles[i])
        
        # Gripper fingers
        p.setJointMotorControl2(robot_id, finger_1, p.POSITION_CONTROL, finger_pos, force=100)
        p.setJointMotorControl2(robot_id, finger_2, p.POSITION_CONTROL, finger_pos, force=100)
        
        p.stepSimulation()
        time.sleep(1./240.)

# --- START THE MISSION ---

print("\n🚀 Mission Start!")

# PHASE 1: Open gripper and move above trash
print("Phase 1: Approaching trash...")
control_robot(target_ee_pos=[0.5, 0, 0.2], finger_pos=0.04, duration_seconds=1.5)

# PHASE 2: Lower arm to touch trash
print("Phase 2: Positioning for grab...")
control_robot(target_ee_pos=[0.5, 0, 0.02], finger_pos=0.04, duration_seconds=1.0)

# PHASE 3: Squeeze!
print("Phase 3: Grabbing trash...")
control_robot(target_ee_pos=[0.5, 0, 0.02], finger_pos=0.01, duration_seconds=1.0)

# PHASE 4: Lift up
print("Phase 4: Lifting...")
control_robot(target_ee_pos=[0.5, 0, 0.3], finger_pos=0.01, duration_seconds=1.5)

# PHASE 5: Move to bin
print("Phase 5: Moving to bin...")
control_robot(target_ee_pos=[0, 0.5, 0.3], finger_pos=0.01, duration_seconds=2.0)

# PHASE 6: Drop!
print("Phase 6: Dropping trash in bin...")
control_robot(target_ee_pos=[0, 0.5, 0.2], finger_pos=0.04, duration_seconds=1.0)

# PHASE 7: Home position
print("Phase 7: Returning home. Mission accomplished!")
control_robot(target_ee_pos=[0.3, 0, 0.4], finger_pos=0.04, duration_seconds=1.5)

time.sleep(1)
p.disconnect()
print("\n🏆 The environment is clean again! Great job!")
