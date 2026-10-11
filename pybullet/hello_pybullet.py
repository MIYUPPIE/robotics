import pybullet as p
import pybullet_data
import time

print("Starting PyBullet in GUI mode...")

# Connect to simulator with GUI
physicsClient = p.connect(p.GUI)

# Set search path for URDFs & plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load ground plane
plane = p.loadURDF("plane.urdf")

# Load a robot (example: Kuka iiwa from Bullet data)
robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0.1], useFixedBase=True)

# Set gravity
p.setGravity(0, 0, -9.81)

print("Applying position control to joint 3 (moving to 90 degrees)...")
# Simple position control example (move joint 3 slowly)
p.setJointMotorControl2(
    robot, 
    3,                      # joint index
    p.POSITION_CONTROL,
    targetPosition=1.57,    # ~90 degrees
    force=300,
    maxVelocity=1.0
)

print("Running simulation loop for 10 seconds. Check the window!")
# Run simulation loop (2400 steps @ 240Hz = 10 econds)
for i in range(2400):
    p.stepSimulation()
    time.sleep(1./240.)     # ~240 Hz real-time

p.disconnect()
print("Simulation ended.")
