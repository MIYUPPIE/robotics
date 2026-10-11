import pybullet as p
import pybullet_data
import time

print("Starting Step 6: Magic Reaching (Inverse Kinematics)!")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Load the robot arm
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

# 1. Choose a target coordinate in the air
target_x = 0.4
target_y = 0.4
target_z = 0.5
target_position = [target_x, target_y, target_z]

# Let's put a tiny red visual ball at that spot so we can see what the robot is reaching for!
marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
p.createMultiBody(baseVisualShapeIndex=marker_visual, basePosition=target_position)

print(f"Goal: Make the fingertip reach the red ball at {target_position}!")

for i in range(1200): # Run for 5 seconds
    p.stepSimulation()
    
    # --- The Magic Math (Inverse Kinematics) ---
    # We ask the engine: "What angles do my 7 joints need to be to make Link #6 touch the red ball?"
    end_effector_index = 6
    calculated_joint_angles = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=end_effector_index,
        targetPosition=target_position
    )
    
    # Now we tell all 7 motors to move to those magically calculated angles!
    for joint_index in range(7):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=calculated_joint_angles[joint_index],
            force=200
        )
        
    time.sleep(1./240.)

p.disconnect()
print("Reached the target!")
