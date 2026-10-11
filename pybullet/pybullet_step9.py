import pybullet as p
import pybullet_data
import time

print("Starting Step 9: Interactive Control (GUI Sliders)!")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Load our favorite Kuka arm
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

# 1. Create Sliders in the PyBullet window
# addUserDebugParameter(name, rangeMin, rangeMax, initialValue)
joint_sliders = []
num_joints = p.getNumJoints(robot_id)

print(f"Adding sliders for {num_joints} joints...")
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode("utf-8")
    
    # We create a slider for each joint!
    slider_id = p.addUserDebugParameter(joint_name, -3.14, 3.14, 0)
    joint_sliders.append(slider_id)

print("\n--- Move the sliders in the 'Parameters' tab on the right! ---")

# 2. Loop and read the sliders
try:
    while True:
        p.stepSimulation()
        
        for i in range(num_joints):
            # Read the value from the slider
            target_pos = p.readUserDebugParameter(joint_sliders[i])
            
            # Tell the joint to move to that position
            p.setJointMotorControl2(
                robot_id, 
                i, 
                p.POSITION_CONTROL, 
                target_pos
            )
            
        time.sleep(1./240.)
except KeyboardInterrupt:
    pass

p.disconnect()
print("Interactive session ended.")
