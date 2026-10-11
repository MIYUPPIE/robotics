import pybullet as p
import pybullet_data
import time

print("Starting Step 5: Driving a Racecar!")

# 1. Start the world
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")

# 2. Load the Racecar!
# We start it slightly in the air so it doesn't get stuck in the floor
car_pos = [0, 0, 0.1]
car_id = p.loadURDF("racecar/racecar.urdf", car_pos)

# 3. Figure out which joints are the wheels
# The racecar has many joints (suspension, steering, etc.). 
# By reading the PyBullet documentation, we know the wheels we want to spin 
# to go forward are Joint #2 (Back Left), #3 (Back Right), 
# #5 (Front Left), and #7 (Front Right).
wheel_joints = [2, 3, 5, 7]
steering_joints = [4, 6] # Front wheels used for turning

# Move the camera so we can see the car better
p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0])

print("\n--- 3... 2... 1... GO! ---")

for i in range(1200): # Run for 5 seconds
    p.stepSimulation()
    
    # Let's make the car drive forward for the first 2 seconds
    if i < 480:
        target_velocity = 20.0 # Spin the wheels fast!
        steering_angle = 0.0   # Go straight
        
    # After 2 seconds, let's turn the steering wheel!
    else:
        target_velocity = 15.0 # Slow down a bit
        steering_angle = 0.5   # Turn the wheels to the right
        
    # --- COMMAND THE CAR ---
    
    # 1. Spin the wheels! We use VELOCITY control instead of POSITION control
    for wheel in wheel_joints:
        p.setJointMotorControl2(
            car_id, 
            wheel, 
            p.VELOCITY_CONTROL, 
            targetVelocity=target_velocity, 
            force=10
        )
        
    # 2. Turn the steering wheel! We use POSITION control
    for hinge in steering_joints:
        p.setJointMotorControl2(
            car_id,
            hinge,
            p.POSITION_CONTROL,
            targetPosition=steering_angle
        )
        
    time.sleep(1./240.)

p.disconnect()
print("Race Finished!")
