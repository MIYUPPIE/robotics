import pybullet as p
import pybullet_data
import time
import math

print("Starting Step 4: Adding a Camera (Robot Eyes)!")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
box_id = p.loadURDF("cube_small.urdf", [0.6, 0, 0.05])

print("\n--- 5. TAKING A PHOTO ---")

# 1. Where do we want to put the camera in the sky?
camera_target_position = [0.6, 0, 0] # Point the camera at the box on the floor
camera_distance = 1.0                # How far away is the camera from the box?
camera_pitch = -45                   # Angle looking downwards
camera_yaw = 45                      # Angle spinning around it
camera_roll = 0                      # Don't tilt the camera sideways

# 2. PyBullet needs to calculate the complicated math for the camera lens
view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=camera_target_position,
    distance=camera_distance,
    yaw=camera_yaw,
    pitch=camera_pitch,
    roll=camera_roll,
    upAxisIndex=2 # Z-axis is up
)

# 3. How wide is the camera lens? (Like zooming in or out)
projection_matrix = p.computeProjectionMatrixFOV(
    fov=60.0,            # Field of View (like a human eye)
    aspect=1.0,          # Square picture
    nearVal=0.1,         # How close can the camera see?
    farVal=100.0         # How far away can the camera see?
)

# 4. SWING THE ARM AND TAKE PICTURES!
p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=1.5, force=500)

for i in range(240): # Run for 1 second
    p.stepSimulation()
    
    # Take a picture every roughly 1/4 second
    if i % 60 == 0:
        print(f"📸 SNAP! Taking a picture at time {i/240:.2f}...")
        
        # This function actually "clicks the shutter" and grabs the pixels!
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=224, 
            height=224,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        # Note: rgb_img is now a massive list of pixel numbers! In real robots, 
        # we would feed this list of pixels into an AI brain.
        
    time.sleep(1./240.)

p.disconnect()
print("Done snapping photos.")
