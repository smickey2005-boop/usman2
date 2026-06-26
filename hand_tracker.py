import cv2
import subprocess
import numpy as np
import mediapipe as mp
from adafruit_servokit import ServoKit

# 1. Initialize PCA9685 PWM hardware
kit = ServoKit(channels=16)
for i in range(6):
    kit.servo[i].set_pulse_width_range(500, 2500)

TENDON_RELEASE = 170  # Open finger angle
TENDON_PULL    = 10   # Closed finger angle

# --- SMOOTHING FILTER CONFIGURATION ---
smoothed_angles = {1: TENDON_RELEASE, 2: TENDON_RELEASE, 3: TENDON_RELEASE, 4: TENDON_RELEASE, 5: TENDON_RELEASE}
ALPHA = 0.25  

def smooth_angle(channel, target_angle):
    current = smoothed_angles[channel]
    new_angle = current + ALPHA * (target_angle - current)
    smoothed_angles[channel] = new_angle
    return max(10, min(170, int(new_angle)))

# 2. Initialize MediaPipe Hand Engine & Drawing Utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print("Starting Hand-Tracking with Skeletal Lines... Press 'q' to exit.")

# Open stable raw video stream pipe via rpicam-vid
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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                # --- DRAW THE SKELETAL TRACKING LINES ---
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # 1. Base Wrist Rotation (Servo 0)
                wrist_x = lm[0].x * 320
                base_angle = 10 + (wrist_x / 320.0) * (170 - 10)
                kit.servo[0].angle = max(10, min(170, int(base_angle)))

                # 2. Individual Finger Control Logic
                t_target = TENDON_RELEASE if lm[4].x > lm[3].x else TENDON_PULL     
                i_target = TENDON_RELEASE if lm[8].y < lm[6].y else TENDON_PULL     
                m_target = TENDON_RELEASE if lm[12].y < lm[10].y else TENDON_PULL   
                r_target = TENDON_RELEASE if lm[16].y < lm[14].y else TENDON_PULL   
                p_target = TENDON_RELEASE if lm[20].y < lm[18].y else TENDON_PULL   

                # 3. Apply smoothing filter
                kit.servo[1].angle = smooth_angle(1, t_target)
                kit.servo[2].angle = smooth_angle(2, i_target)
                kit.servo[3].angle = smooth_angle(3, m_target)
                kit.servo[4].angle = smooth_angle(4, r_target)
                kit.servo[5].angle = smooth_angle(5, p_target)

        else:
            for ch in range(1, 6):
                kit.servo[ch].angle = smooth_angle(ch, TENDON_RELEASE)

        cv2.imshow("FYP Bionic Hand - MediaPipe Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam_process.terminate()
    cv2.destroyAllWindows()
    hands.close()
