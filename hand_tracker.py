import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
from adafruit_servokit import ServoKit

# Initialize PCA9685 PWM board
kit = ServoKit(channels=16)
for i in range(6):
    kit.servo[i].set_pulse_width_range(500, 2500)

TENDON_RELEASE = 170  # Open finger angle
TENDON_PULL    = 10   # Closed finger angle

# --- SMOOTHING CONFIGURATION ---
# Keeps track of the last sent angle to prevent sudden drops/twitches
smoothed_angles = {1: 170, 2: 170, 3: 170, 4: 170, 5: 170}
ALPHA = 0.3  # Smoothing factor (0.1 = ultra slow/smooth, 1.0 = instant/harsh)

def smooth_angle(channel, target_angle):
    """Calculates a moving average transition to eliminate jitter."""
    current = smoothed_angles[channel]
    new_angle = current + ALPHA * (target_angle - current)
    smoothed_angles[channel] = new_angle
    return max(10, min(170, int(new_angle)))
# -------------------------------

# Load the dedicated Hand-Pose Tracking Model
model = YOLO('yolov8n-pose-hand.pt') 

print("Starting Jitter-Free 5-Finger Tracking Loop... Press 'q' to exit.")

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

        results = model(frame, stream=True, conf=0.35, verbose=False)
        hand_detected = False

        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                joints = r.keypoints.xy[0].cpu().numpy()
                
                if len(joints) >= 21:
                    hand_detected = True
                    wrist_x, wrist_y = joints[0][0], joints[0][1]
                    
                    # 1. Smooth Base Wrist Side Rotation (Servo 0)
                    base_target = 10 + (wrist_x / 320.0) * (170 - 10)
                    kit.servo[0].angle = max(10, min(170, int(base_target)))

                    # 2. Determine target raw binary positions
                    t_target = TENDON_RELEASE if joints[4][0] > joints[3][0] else TENDON_PULL
                    i_target = TENDON_RELEASE if joints[8][1] < joints[6][1] else TENDON_PULL
                    m_target = TENDON_RELEASE if joints[12][1] < joints[10][1] else TENDON_PULL
                    r_target = TENDON_RELEASE if joints[16][1] < joints[14][1] else TENDON_PULL
                    p_target = TENDON_RELEASE if joints[20][1] < joints[18][1] else TENDON_PULL

                    # 3. Apply smoothing filter before setting hardware pulse
                    kit.servo[1].angle = smooth_angle(1, t_target)
                    kit.servo[2].angle = smooth_angle(2, i_target)
                    kit.servo[3].angle = smooth_angle(3, m_target)
                    kit.servo[4].angle = smooth_angle(4, r_target)
                    kit.servo[5].angle = smooth_angle(5, p_target)

                    print(f"Smoothed -> T: {smoothed_angles[1]:.0f} | I: {smoothed_angles[2]:.0f} | M: {smoothed_angles[3]:.0f}")

        # If hand leaves frame entirely, gently return servos to default open position
        if not hand_detected:
            for ch in range(1, 6):
                kit.servo[ch].angle = smooth_angle(ch, TENDON_RELEASE)

        cv2.imshow("FYP Bionic Hand - AI Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam_process.terminate()
    cv2.destroyAllWindows()
