import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
from adafruit_servokit import ServoKit

# Initialize all 16 channels on the PCA9685 board
kit = ServoKit(channels=16)

# Setup all 6 active servos (Channels 0 to 5)
for i in range(6):
    kit.servo[i].set_pulse_width_range(500, 2500)

# Define safe mechanical limits for your fingers (adjust based on your string/wire tension)
FINGER_CLOSE = 10    # Angle when finger is pulled tight / closed
FINGER_OPEN = 170    # Angle when finger is released / wide open

# Load tracking model
model = YOLO('yolov8n-pose.pt')

print("Bionic Hand 6-Servo System Active... Press 'q' to exit.")

# Open rpicam-vid stream
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
                if len(joints) > 5: # Make sure upper body/hand keypoints are visible
                    
                    # 1. Wrist Base Rotation (Servo 0) - Track side-to-side
                    wrist_x = joints[0][0]
                    base_angle = 10 + (wrist_x / 320.0) * (170 - 10)
                    kit.servo[0].angle = max(10, min(170, base_angle))

                    # 2. Finger Gestures Logic (Servos 1 to 5)
                    # We compare the height of upper points vs lower points to see if hand is open
                    point_high = joints[1][1]  # Higher point up the arm/hand
                    point_low = joints[0][1]   # Lower point (wrist)
                    
                    # If the distance is large, your hand is extended open!
                    if (point_low - point_high) > 35:
                        print("Hand State: OPEN! -> Extending all fingers.")
                        kit.servo[1].angle = FINGER_OPEN  # Thumb
                        kit.servo[2].angle = FINGER_OPEN  # Index
                        kit.servo[3].angle = FINGER_OPEN  # Middle
                        kit.servo[4].angle = FINGER_OPEN  # Ring
                        kit.servo[5].angle = FINGER_OPEN  # Pinky
                    else:
                        print("Hand State: CLOSED! -> Clenching fist.")
                        kit.servo[1].angle = FINGER_CLOSE
                        kit.servo[2].angle = FINGER_CLOSE
                        kit.servo[3].angle = FINGER_CLOSE
                        kit.servo[4].angle = FINGER_CLOSE
                        kit.servo[5].angle = FINGER_CLOSE

                    # Visual feedback dot on screen
                    cv2.circle(frame, (int(wrist_x), int(joints[0][1])), 8, (0, 255, 0), -1)

        cv2.imshow("FYP Bionic Hand - AI Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam_process.terminate()
    cv2.destroyAllWindows()
