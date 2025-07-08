import cv2
import mediapipe as mp

#Initialize objects
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#Config detection settings
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

#Fetch a source
cap = cv2.VideoCapture('sample.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        print('Video ended')
        break
    
    #Convert frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Process frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    resized = cv2.resize(frame, (1820,980))

    cv2.imshow('output',resized)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break