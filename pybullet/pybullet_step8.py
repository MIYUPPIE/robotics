import pybullet as p
import pybullet_data
import time

print("Starting Step 8: Physics Playground (Friction & Mass)!")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# 1. Create a "Heavy" Box
# We spend a bit more time on 'changeDynamics' here
heavy_box_id = p.loadURDF("cube_small.urdf", [0, 0, 1])
p.changeDynamics(heavy_box_id, -1, mass=10.0) # -1 means the base of the object
print("Created a HEAVY box (10kg)!")

# 2. Create a "Slippery" Box
slippery_box_id = p.loadURDF("cube_small.urdf", [0.5, 0, 1])
p.changeDynamics(slippery_box_id, -1, lateralFriction=0.01)
print("Created a SLIPPERY box (Low friction)!")

# 3. Create a "Bouncy" Box
bouncy_box_id = p.loadURDF("cube_small.urdf", [-0.5, 0, 1])
p.changeDynamics(bouncy_box_id, -1, restitution=0.9)
print("Created a BOUNCY box (High restitution)!")

# 4. Let's make a mini slope to test the friction!
# We'll use a simple visual shape for a ramp
ramp_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05], rgbaColor=[0.7, 0.7, 0.7, 1])
ramp_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05])
# Tilt the ramp by 30 degrees (0.5 radians)
ramp_id = p.createMultiBody(
    baseMass=0, # Fixed in place
    baseCollisionShapeIndex=ramp_collision,
    baseVisualShapeIndex=ramp_visual,
    basePosition=[0, 1, 0.2],
    baseOrientation=p.getQuaternionFromEuler([0.5, 0, 0])
)

# Put the slippery box on the ramp!
p.resetBasePositionAndOrientation(slippery_box_id, [0, 1, 0.5], p.getQuaternionFromEuler([0.5, 0, 0]))

# Put a normal box on the ramp to compare!
normal_box_id = p.loadURDF("cube_small.urdf", [0.2, 1, 0.5])
p.resetBasePositionAndOrientation(normal_box_id, [0.2, 1, 0.5], p.getQuaternionFromEuler([0.5, 0, 0]))

print("\n--- Watch them fall and slide! ---")
for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
print("Physics lesson finished!")
