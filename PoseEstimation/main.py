import cv2
import mediapipe as mp

#Load objects
mp_pose = mp.solutions.pose
drawing = mp.solutions.drawing_utils

#Load video
cap = cv2.VideoCapture('sample.mp4')

with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:

    while True:
        ret, frame = cap.read()

        if not ret:
            print('Video ended')
            break

        #Detect pose
        results = pose.process(frame)

        #Draw Results

        if results.pose_landmarks:
            drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=3))
        

        display_frame = cv2.resize(frame, (1820,980))
        cv2.imshow('Output',display_frame)
        
        #Adjust clip speed
        key = cv2.waitKey(20)

        if key == ord('q'):
            break
