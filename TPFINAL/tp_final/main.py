import pickle
import cv2
import mediapipe as mp
import numpy as np
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load sound mappings
sound_files = {
    'a': './piano/do-stretched.wav',
    'b': './piano/re-stretched.wav',
    'c': './piano/mi-stretched.wav',
    'd': './piano/fa-stretched.wav',
    'i': './piano/sol-stretched.wav',
}

sound_files_low_pitch = {
    'a': './piano/low-pitch/do-stretched_low_pitch.mp3',
    'b': './piano/low-pitch/re-stretched_low_pitch.mp3',
    'c': './piano/low-pitch/mi-stretched_low_pitch.mp3',
    'd': './piano/low-pitch/fa-stretched_low_pitch.mp3',
    'i': './piano/low-pitch/sol-stretched_low_pitch.mp3',
}

sound_files_high_pitch = {
    'a': './piano/high-pitch/do-stretched_high_pitch.mp3',
    'b': './piano/high-pitch/re-stretched_high_pitch.mp3',
    'c': './piano/high-pitch/mi-stretched_high_pitch.mp3',
    'd': './piano/high-pitch/fa-stretched_high_pitch.mp3',
    'i': './piano/high-pitch/sol-stretched_high_pitch.mp3',
}


# Pre-load sound objects for efficiency
sounds = {key: pygame.mixer.Sound(file) for key, file in sound_files.items()}
sounds_low_pitch = {key: pygame.mixer.Sound(file) for key, file in sound_files_low_pitch.items()}
sounds_high_pitch = {key: pygame.mixer.Sound(file) for key, file in sound_files_high_pitch.items()}


# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)

# Define the labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 20: 'U', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
               13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Flag to track if the sound has been played for each key
sound_played = {key: False for key in sound_files.keys()}

# Function to play piano sound
# Function to play piano sound
def tocar_piano(predicted_character, tilt):
    global sound_played  # Declare sound_played as global to retain state
    character = predicted_character.lower()
    
    # Select the appropriate sound mapping based on tilt
    if tilt == "upwards":
        current_sounds = sounds_high_pitch
    elif tilt == "downwards":
        current_sounds = sounds_low_pitch
    else:  # Neutral tilt
        current_sounds = sounds

    if character in current_sounds:
        if not sound_played[character]:
            current_sounds[character].play()
            sound_played[character] = True
    
    # Reset the flag for other characters
    for key in sound_played.keys():
        if key != character:
            sound_played[key] = False
    

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty frame.")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape

    # Hand Detection
    hand_results = hands.process(frame_rgb)
    data_aux = []
    x_ = []
    y_ = []

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Play piano sound with tilt
            tocar_piano(predicted_character, head_tilt_flag)

            # Display prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        except Exception as e:
            pass


    # Face Mesh Detection
    face_results = face_mesh.process(frame_rgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]  # Nose tip
            chin = face_landmarks.landmark[152]   # Chin
            forehead = face_landmarks.landmark[10]  # Forehead

            # Get screen space coordinates
            image_height, image_width, _ = frame.shape
            nose_y = nose_tip.y * image_height
            chin_y = chin.y * image_height
            forehead_y = forehead.y * image_height

            face_height = abs(forehead_y - chin_y)
            if face_height == 0:
                continue

            normalized_tilt = (chin_y - nose_y) / face_height

            # Update head tilt flag
            if normalized_tilt > 0.5:
                head_tilt_flag = "upwards"
            elif normalized_tilt < 0.4:
                head_tilt_flag = "downwards"
            else:
                head_tilt_flag = "neutral"

            # Draw landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Display head tilt flag
            cv2.putText(frame, f"Head Tilt: {head_tilt_flag}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand and Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
