import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
from adafruit_servokit import ServoKit

# 1. Initialize PCA9685 PWM board
kit = ServoKit(channels=16)
for i in range(6):
    kit.servo[i].set_pulse_width_range(500, 2500)

TENDON_RELEASE = 170  # Open finger angle
TENDON_PULL    = 10   # Closed finger angle

# 2. Load dedicated Hand-Pose Tracking Model (Isolates 21 hand points)
# This will automatically download a specialized model configured for precise finger locations
model = YOLO('yolov8n-pose-hand.pt') 

print("Starting True 5-Finger Tracking Engine... Press 'q' to exit.")

# Open video stream pipe via rpicam-vid
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
                
                # Check if all 21 keypoints of the hand structure are tracked
                if len(joints) >= 21:
                    # Anchor Point: Wrist Base position
                    wrist_x, wrist_y = joints[0][0], joints[0][1]
                    
                    # 1. Base Wrist Side Rotation (Servo 0)
                    base_angle = 10 + (wrist_x / 320.0) * (170 - 10)
                    kit.servo[0].angle = max(10, min(170, base_angle))

                    # 2. Individual Finger Extension Logic (Comparing Tip heights vs Knuckle bases)
                    # Hand Point Map: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
                    thumb_state  = TENDON_RELEASE if joints[4][0] > joints[3][0] else TENDON_PULL
                    index_state  = TENDON_RELEASE if joints[8][1] < joints[6][1] else TENDON_PULL
                    middle_state = TENDON_RELEASE if joints[12][1] < joints[10][1] else TENDON_PULL
                    ring_state   = TENDON_RELEASE if joints[16][1] < joints[14][1] else TENDON_PULL
                    pinky_state  = TENDON_RELEASE if joints[20][1] < joints[18][1] else TENDON_PULL

                    # Write target angles directly to the hardware pins
                    kit.servo[1].angle = thumb_state
                    kit.servo[2].angle = index_state
                    kit.servo[3].angle = middle_state
                    kit.servo[4].angle = ring_state
                    kit.servo[5].angle = pinky_state

                    print(f"Fingers -> T: {thumb_state} | I: {index_state} | M: {middle_state} | R: {ring_state} | P: {pinky_state}")

                    # Render tracking markers onto the active stream window
                    for pt in [4, 8, 12, 16, 20]: # Draw dots on fingertips
                        cv2.circle(frame, (int(joints[pt][0]), int(joints[pt][1])), 5, (0, 255, 0), -1)

        cv2.imshow("FYP Bionic Hand - AI Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam_process.terminate()
    cv2.destroyAllWindows()
