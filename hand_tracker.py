import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
from adafruit_servokit import ServoKit

# Initialize the 16-channel PCA9685 board
# This connects over I2C automatically
kit = ServoKit(channels=16)

# Set up Servo 0 (Base rotation example)
# Adjust min_pulse and max_pulse if your servos need exact calibration
kit.servo[0].set_pulse_width_range(500, 2500)

# Load the downloaded hand-pose model
model = YOLO('yolov8n-pose.pt')

print("AI Tracking + Servo Controller Initialized... Press 'q' to exit.")

# Open rpicam-vid pipe
cmd = [
    'rpicam-vid', '-t', '0',
    '--width', '320', '--height', '240',
    '--framerate', '30', '--codec', 'yuv420', '-o', '-'
]
cam_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**6)
frame_size = int(320 * 240 * 1.5)

try:
    while True:
        raw_frame = cam_process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            break

        yuv_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((int(240 * 1.5), 320))
        frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        frame = cv2.flip(frame, 1)

        results = model(frame, stream=True, conf=0.4, verbose=False)

        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                joints = r.keypoints.xy[0].cpu().numpy()
                if len(joints) > 0:
                    # Capture tracked wrist coordinate
                    wrist_x = joints[0][0]
                    
                    # --- LINEAR MAP EQUATION ---
                    # Map pixel space (0 to 320) to physical angles (10 to 170 degrees)
                    # Safe limits protect the physical 3D printed/acrylic arm joints
                    target_angle = 10 + (wrist_x / 320.0) * (170 - 10)
                    target_angle = max(10, min(170, target_angle)) # Bound check
                    
                    print(f"Wrist X: {wrist_x:.1f}px ---> Writing Servo 0 Angle: {target_angle:.1f}°")
                    
                    # Direct hardware write to PCA9685 channel 0
                    kit.servo[0].angle = target_angle
                    
                    # Draw indicator point on feed
                    cv2.circle(frame, (int(wrist_x), int(joints[0][1])), 8, (0, 255, 0), -1)

        cv2.imshow("FYP Robotic Arm - AI Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam_process.terminate()
    cv2.destroyAllWindows()