import pybullet as p
import pybullet_data
import time

print("Starting Step 3: Collisions & Touch!")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Set up the floor and robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

# Put a tiny box on the floor right in front of the robot
box_id = p.loadURDF("cube_small.urdf", [0.6, 0, 0.05])

print("\n--- Let the box settle on the floor ---")
for _ in range(240): # 1 second
    p.stepSimulation()
    time.sleep(1./240.)

print("\n--- 4. DETECTING COLLISIONS (BUMP!) ---")
print("Swinging the robot arm down towards the box...")

# Joint 1 is the shoulder that bends the arm forward/down
p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=1.5, force=300)

already_hit = False

for i in range(720): # 3 seconds
    p.stepSimulation()
    
    # We ask the physics engine: "Are the robot and the box touching each other?"
    # It returns a list of contact points. Empty list = not touching.
    contacts = p.getContactPoints(robot_id, box_id)
    
    if len(contacts) > 0 and not already_hit:
        print(f"💥 SMASH! The robot touched the box at {i/240:.1f} seconds!")
        already_hit = True # So we only print it once
        
    time.sleep(1./240.)

p.disconnect()
print("Done.")
