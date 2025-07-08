import cv2
import mediapipe as mp

#Initialize Objects
mp_face_mash = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

#Face mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#Configure Mesh
face_mash = mp_face_mash.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#Load video source
cap = cv2.VideoCapture('sample.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        print('Video ended')
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Extract landmarks
    results = face_mash.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mash.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )

    resized = cv2.resize(frame, (1820,980))
    cv2.imshow('Output',resized)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
