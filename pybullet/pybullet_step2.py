import pybullet as p
import pybullet_data
import time

# Start simulator
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load environment
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

print("\n--- 1. EXPLORING THE ROBOT'S JOINTS ---")
num_joints = p.getNumJoints(robot_id)
print(f"The robot has {num_joints} joints total (including fixed joints).")

for j in range(num_joints):
    # getJointInfo returns a tuple of 17 variables. We care about the name and type.
    info = p.getJointInfo(robot_id, j)
    joint_index = info[0]
    joint_name = info[1].decode('utf-8') # It's a byte string by default
    joint_type = info[2]                 # 0=REVOLUTE, 1=PRISMATIC, 4=FIXED
    
    type_str = "REVOLUTE (Motor)" if joint_type == 0 else "FIXED" if joint_type == 4 else str(joint_type)
    print(f"  Index {joint_index}: {joint_name} | Type: {type_str}")

print("\n--- 2. ADDING OBJECTS INTO THE WORLD ---")
# Let's drop a small cube in front of the robot (x=0.5m, y=0m, z=1.0m)
cube_start_pos = [0.5, 0, 1.0]
cube_id = p.loadURDF("cube_small.urdf", cube_start_pos)
print("Dropped a tiny box from the sky!")

print("\n--- 3. READING ROBOT SENSORS/STATES ---")
# Link 6 is usually the end-effector (the tip of the arm) for the Kuka iiwa
end_effector_index = 6

# Let's move joint 1 just to see the arm swing and update states
p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=1.0, force=200)

for i in range(1200): # 5 seconds
    p.stepSimulation()
    
    # A. Read a specific joint's angle
    # returns: (jointPosition, jointVelocity, jointReactionForces, appliedJointMotorTorque)
    joint_state = p.getJointState(robot_id, 1)
    joint_pos = joint_state[0]
    
    # B. Read a specific link's 3D coordinates in the world
    # returns: (linkWorldPosition, linkWorldOrientation, ...)
    link_state = p.getLinkState(robot_id, end_effector_index)
    link_pos = link_state[0] # (x, y, z)
    
    # Print every 240 steps (1 second of sim time)
    if i % 240 == 0:
        print(f"Time {i/240:.1f}s -> Joint 1 Angle: {joint_pos:.2f} rad | "
              f"End-Effector XYZ: ({link_pos[0]:.2f}, {link_pos[1]:.2f}, {link_pos[2]:.2f})")
        
    time.sleep(1./240.)

p.disconnect()
print("Done.")
