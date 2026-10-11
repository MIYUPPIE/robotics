import pybullet as p
import pybullet_data
import time

print("Starting Step 7: Picking Up Objects (Grasping)!")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# 1. Load a robot with a REAL gripper! (The Franka Panda)
# We put it at [0, 0, 0]
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

# 2. Put a block on the table (or floor)
block_pos = [0.5, 0, 0.05]
block_id = p.loadURDF("cube_small.urdf", block_pos)

# Standard index for Panda end-effector is 11, and gripper fingers are 9 and 10
ee_index = 11
finger_1 = 9
finger_2 = 10

def move_arm(target_pos):
    # Use IK to find the angles
    joint_angles = p.calculateInverseKinematics(robot_id, ee_index, target_pos)
    # Move the 7 arm joints (0 to 6)
    for i in range(7):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_angles[i])

print("\n--- Phase 1: Reach for the block ---")
# Use our IK magic from Step 6 to calculate angles for the block's position
# We aim slightly above the block first
target_pos = [0.5, 0, 0.2]

for i in range(240):
    joint_angles = p.calculateInverseKinematics(robot_id, ee_index, [0.5, 0, 0.05])
    # Apply to joints 0-8 (Panda has more joints)
    for j in range(9):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, joint_angles[j])
    p.stepSimulation()
    time.sleep(1./240.)

print("\n--- Phase 2: Close the fingers! ---")
# Joints 9 and 10 are the fingers. 0.0 means closed, 0.04 means open.
for i in range(240):
    p.setJointMotorControl2(robot_id, finger_1, p.POSITION_CONTROL, 0.01, force=100)
    p.setJointMotorControl2(robot_id, finger_2, p.POSITION_CONTROL, 0.01, force=100)
    p.stepSimulation()
    time.sleep(1./240.)

print("\n--- Phase 3: Lift it up! ---")
for i in range(480):
    # Keep the fingers squeezed tight!
    p.setJointMotorControl2(robot_id, finger_1, p.POSITION_CONTROL, 0.01, force=100)
    p.setJointMotorControl2(robot_id, finger_2, p.POSITION_CONTROL, 0.01, force=100)
    
    # Lift the arm straight up
    joint_angles = p.calculateInverseKinematics(robot_id, ee_index, [0.5, 0, 0.5])
    for j in range(9):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, joint_angles[j])
        
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
print("Look! We picked up the block!")
