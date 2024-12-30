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
    'e': './piano/sol-stretched.wav',
}

# Pre-load sound objects for efficiency
sounds = {key: pygame.mixer.Sound(file) for key, file in sound_files.items()}

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start video capture
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# Flag to track if the sound has been played for each key
sound_played = {key: False for key in sound_files.keys()}

# Function to play piano sound
def tocar_piano(predicted_character):
    global sound_played  # Declare sound_played as global to retain state
    character = predicted_character.lower()
    if character in sounds:
        if not sound_played[character]:
            sounds[character].play()
            sound_played[character] = True
    for key in sound_played.keys():
        if key != character:
            sound_played[key] = False  # Reset the flag for other characters

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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
        print("Predicted character : ", predicted_character)

        tocar_piano(predicted_character)  # Call the piano function

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    except Exception as e:
        pass

    cv2.imshow('frame', frame)

    # Check if the window was closed by the user (pressing 'q' key)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()