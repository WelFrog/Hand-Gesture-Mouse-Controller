import cv2
import mediapipe as mp
import ctypes
import math
import pyautogui
import time

mouse_controller = ctypes.CDLL('MouseController.dll')

print("Import .dll success.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

print(screen_width, screen_height)
print("Init success.")

cap = cv2.VideoCapture(0)

print("Open camara success.")

f = 0
is_moving = False
prev_wrist_pos = None  # 记录上一帧的手腕位置
velocity = [0, 0]  # 鼠标速度
alpha = 0.8  # 平滑参数
acceleration_factor = 2.0  # 加速因子
sticky_mode = False  # 粘滞模式标志
press_start_time = None  # 按下开始时间
release_start_time = None  # 分开开始时间
operation_hand_wrist = None  # 当前操作手腕的位置

def calc_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    min_dist = float('inf')  # 用于选择最接近上一状态手腕的手
    selected_hand = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 获取关键点
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # 选择最接近上一操作手腕的手
            if operation_hand_wrist is not None:
                dist_to_prev = calc_distance(wrist, operation_hand_wrist)
                if dist_to_prev < min_dist:
                    min_dist = dist_to_prev
                    selected_hand = hand_landmarks
            else:
                selected_hand = hand_landmarks
                break  # 如果之前没有操作手腕，则选第一个检测到的手

        if selected_hand:
            operation_hand_wrist = selected_hand.landmark[mp_hands.HandLandmark.WRIST]  # 更新操作手的手腕位置

            finger_tips = [
                selected_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],   # 食指
                selected_hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],  # 中指
                selected_hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],    # 无名指
                selected_hand.landmark[mp_hands.HandLandmark.PINKY_TIP],          # 小指
                selected_hand.landmark[mp_hands.HandLandmark.THUMB_TIP],          # 大拇指
                operation_hand_wrist,                                             # 手腕
            ]

            wrist = finger_tips[5]
            ring_finger_tip = finger_tips[2]
            pinky_tip = finger_tips[3]

            # 判断是否进入移动状态
            dist_ring_to_wrist = calc_distance(ring_finger_tip, wrist)
            dist_pinky_to_wrist = calc_distance(pinky_tip, wrist)

            if dist_ring_to_wrist < 0.2 and dist_pinky_to_wrist < 0.2:
                is_moving = True
            else:
                is_moving = False
                prev_wrist_pos = None  # 重置手腕位置
                velocity = [0, 0]  # 重置速度

            if is_moving:
                # 当前手腕位置
                current_wrist_pos = (wrist.x, wrist.y)

                if prev_wrist_pos:
                    # 计算手腕移动量
                    dx = prev_wrist_pos[0] - current_wrist_pos[0]  # X轴反向
                    dy = current_wrist_pos[1] - prev_wrist_pos[1]

                    # 映射到屏幕分辨率
                    mouse_dx = dx * screen_width * acceleration_factor
                    mouse_dy = dy * screen_height * acceleration_factor

                    # 更新速度（平滑处理）
                    velocity[0] = alpha * velocity[0] + (1 - alpha) * mouse_dx
                    velocity[1] = alpha * velocity[1] + (1 - alpha) * mouse_dy

                    # 移动鼠标
                    mouse_x, mouse_y = pyautogui.position()
                    target_x = mouse_x + velocity[0]
                    target_y = mouse_y + velocity[1]
                    mouse_controller.moveMouse(int(target_x), int(target_y))

                # 更新上一帧的手腕位置
                prev_wrist_pos = current_wrist_pos

            # 捏合和粘滞模式逻辑
            thumb_tip = finger_tips[4]
            index_tip = finger_tips[0]
            dist_thumb_index = calc_distance(thumb_tip, index_tip)

            if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
                if not sticky_mode:
                    if press_start_time is None:
                        press_start_time = time.time()
                        mouse_controller.mouseDown()
                        f = 1
                    elif time.time() - press_start_time > 0.5:
                        sticky_mode = True
                release_start_time = None
            else:
                if sticky_mode:
                    if release_start_time is None:
                        release_start_time = time.time()
                    elif time.time() - release_start_time > 0.3:
                        sticky_mode = False
                        mouse_controller.mouseUp()
                        f = 0
                else:
                    mouse_controller.mouseUp()
                    f = 0
                    press_start_time = None

    # 显示视频流
    cv2.imshow('Hand Gesture Recognition', frame)

cap.release()
cv2.destroyAllWindows()
