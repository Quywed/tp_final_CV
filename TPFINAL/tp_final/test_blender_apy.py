import threading
import logging
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pygame
from ultralytics import YOLO
import time
import numpy as np
import socket
import json
import queue
import subprocess
import os
import random
import tkinter as tk
from tkinter import messagebox

class BlenderAudioSender:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.audio_queue = queue.Queue()
        self.running = True
        self.sock = None
        self.connected = False
        
        # Start sender thread
        self.sender_thread = threading.Thread(target=self._sender_loop)
        self.sender_thread.daemon = True
        self.sender_thread.start()
    
    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.sock.listen()
            print(f"Waiting for Blender connection on {self.host}:{self.port}")
            self.conn, addr = self.sock.accept()
            print(f"Connected to Blender at {addr}")
            self.connected = True
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
    
    def _sender_loop(self):
        self._connect()
        while self.running:
            try:
                if not self.connected:
                    self._connect()
                    continue
                
                # Get audio data from queue
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    data_json = json.dumps(data)
                    self.conn.sendall(data_json.encode() + b'\n')
                except queue.Empty:
                    # Send zero data when no sound is playing
                    zero_data = [0.0] * 256
                    data_json = json.dumps(zero_data)
                    self.conn.sendall(data_json.encode() + b'\n')
                    time.sleep(0.01)
                
            except Exception as e:
                print(f"Sender error: {e}")
                self.connected = False
                time.sleep(0.1)
    
    def send_audio_data(self, audio_data):
        """Add audio data to the queue for sending to Blender"""
        self.audio_queue.put(audio_data)
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.sock:
            self.sock.close()

class AudioCaptureMixer:
    def __init__(self, original_sound):
        self.original_sound = original_sound
        self.sample_rate = 44100  # Standard sample rate
        
    def play(self):
        self.original_sound.play()
        
        # Generate approximate waveform data
        duration = 0.1  # Duration in seconds
        samples = np.linspace(0, duration, 256)
        waveform = np.sin(2 * np.pi * 440 * samples) * 0.5
        
        # Send to Blender
        blender_sender.send_audio_data(waveform.tolist())

# Initialize Blender sender before pygame
blender_sender = BlenderAudioSender()

# Launch visualizer executable
try:
    visualizer_path = "./visualizer.exe"  # Update this path to match your .exe location
    if os.path.exists(visualizer_path):
        subprocess.Popen([visualizer_path])
        print("Visualizer launched successfully")
    else:
        print(f"Warning: Visualizer not found at {visualizer_path}")
except Exception as e:
    print(f"Error launching visualizer: {e}")


# Initialize pygame mixer
pygame.mixer.init()
print("Escreva 'obj' no terminal para iniciar a detecao de objetos, 'stop' para parar.")

backpack_window_open = False
def backpack_sounds():
    global backpack_window_open

    if backpack_window_open:
        return  # If the window is already open, do nothing

    backpack_window_open = True  # Set the flag to indicate the window is open

    def list_sound_files(directory):
        """List all sound files in the given directory."""
        return [f for f in os.listdir(directory) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]

    def play_sound():
        """Play the sound file entered in the textbox."""
        file_name = entry.get().strip()
        if not file_name:
            messagebox.showerror("Error", "Please enter a file name.")
            return
        
        file_path = os.path.join(current_directory, file_name)
        if not os.path.isfile(file_path):
            messagebox.showerror("Error", f"File '{file_name}' not found.")
            return
        
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.set_volume(0.25)  # Set volume to 50%
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror("Error", f"Could not play sound: {str(e)}")

    def stop_sound():
        """Stop any currently playing sound."""
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    def on_file_selected(event):
        """Fill the entry with the selected file name."""
        selection = listbox.curselection()
        if selection:
            entry.delete(0, tk.END)
            entry.insert(0, listbox.get(selection))

    def update_file_list():
        """Update the listbox with sound files from the current directory."""
        listbox.delete(0, tk.END)
        files = list_sound_files(current_directory)
        for file in files:
            listbox.insert(tk.END, file)

    def on_close():
        """Reset the flag when the window is closed."""
        global backpack_window_open
        backpack_window_open = False
        root.destroy()

    current_directory = './custom_music'
    if not os.path.exists(current_directory):
        os.makedirs(current_directory)
    pygame.init()

    # Create the main window
    root = tk.Tk()
    root.title("Sound Player")
    root.protocol("WM_DELETE_WINDOW", on_close)  # Handle window close event

    # Create and pack the widgets
    frame = tk.Frame(root)
    frame.pack(pady=10)

    entry = tk.Entry(frame, width=40)
    entry.grid(row=0, column=0, padx=5)

    play_button = tk.Button(frame, text="Play", command=play_sound)
    play_button.grid(row=0, column=1, padx=5)

    stop_button = tk.Button(frame, text="Stop", command=stop_sound)
    stop_button.grid(row=0, column=2, padx=5)

    listbox = tk.Listbox(root, width=60, height=15)
    listbox.pack(pady=10)
    listbox.bind("<<ListboxSelect>>", on_file_selected)

    update_file_list()

    # Run the Tkinter event loop
    root.mainloop()

def play_random_music():
    music_folder='./custom_music'

    try:
        music_files = [f for f in os.listdir(music_folder) if f.endswith('.mp3') or f.endswith('.wav')]
        if not music_files:
            print(f"No music files found in {music_folder}")
            return
        song = random.choice(music_files)
        song_path = os.path.join(music_folder, song)
        print(f"Playing random song: {song_path}")
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.set_volume(0.15)  # Set volume to 15%
        pygame.mixer.music.play()
        time.sleep(2)
    except Exception as e:
        print(f"Error playing random music: {e}")

def stop_all_sounds():

    pygame.mixer.music.stop()  # Stop any random music playing
    for sound in sounds.values():
        sound.stop()  # Stop all instrument sounds
    for sound in sounds_low_pitch.values():
        sound.stop()
    for sound in sounds_high_pitch.values():
        sound.stop()
    for sound in sounds_bongo.values():
        sound.stop()
    for sound in sounds_drums.values():
        sound.stop()
    print("All sounds stopped.")

def wrap_sound(sound):
    return AudioCaptureMixer(sound)

# [Previous sound file definitions remain the same...]
# Define sound files for different pitches
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

sound_files_bongo = {
    'a': './bongo/congas-conga-open.wav',
    'b': './bongo/lp-bongos-hi.wav',
    'c': './bongo/lp-bongos-hi-slap.wav',
    'd': './bongo/lp-bongos-low.wav',
    'i': './bongo/lp-congas-quinto-muted-slap.wav',
}

sound_files_drums = {
    'a': './drums/closed-hh.wav',
    'b': './drums/bell.wav',
    'c': './drums/kick.wav',
    'd': './drums/rack-tom-1.wav',
    'i': './drums/rack-tom-2.wav',
}
# Load sound objects for different pitches
sounds = {key: wrap_sound(pygame.mixer.Sound(file)) for key, file in sound_files.items()}
sounds_low_pitch = {key: wrap_sound(pygame.mixer.Sound(file)) for key, file in sound_files_low_pitch.items()}
sounds_high_pitch = {key: wrap_sound(pygame.mixer.Sound(file)) for key, file in sound_files_high_pitch.items()}
sounds_bongo = {key: wrap_sound(pygame.mixer.Sound(file)) for key, file in sound_files_bongo.items()}
sounds_drums = {key: wrap_sound(pygame.mixer.Sound(file)) for key, file in sound_files_drums.items()}

metronome_sound = wrap_sound(pygame.mixer.Sound('./metronome/click.wav'))

# Add metronome state variables
metronome_active = False
metronome_thread = None

def play_metronome():
    global metronome_active
    while metronome_active:
        metronome_sound.play()
        time.sleep(0.9)  # 100 BPM = 0.6 seconds between beats

def toggle_metronome():
    global metronome_active, metronome_thread
    
    if not metronome_active:
        metronome_active = True
        metronome_thread = threading.Thread(target=play_metronome)
        metronome_thread.start()
    else:
        metronome_active = False
        if metronome_thread:
            metronome_thread.join()

# Add current instrument tracker
current_instrument = "piano"  # Default instrument

# [Previous model loading and initialization code remains the same...]
# Load ASL model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,min_detection_confidence=0.5, min_tracking_confidence=0.5)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 20: 'U', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
               13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

sound_played = {key: False for key in sound_files.keys()}

def tocar_instrumento(predicted_character, tilt):
    global sound_played
    character = predicted_character.lower()

    if predicted_character.lower() == 'u':
        stop_all_sounds()

    if current_instrument == "bongo":
        current_sounds = sounds_bongo
    elif current_instrument == "drums":
        current_sounds = sounds_drums
    else:  # piano
        if tilt == "cima":
            current_sounds = sounds_high_pitch
        elif tilt == "baixo":
            current_sounds = sounds_low_pitch
        else:
            current_sounds = sounds

    if character in current_sounds:
        if not sound_played[character]:
            current_sounds[character].play()
            sound_played[character] = True
    
    for key in sound_played.keys():
        if key != character:
            sound_played[key] = False

def draw_detections(frame, results):
    """Draw boxes only for bottles and cell phones"""
    annotated_frame = frame.copy()
    
    if not results or not results[0].boxes:
        return annotated_frame
    
    boxes = results[0].boxes
    for box in boxes:
        # Get class ID and name
        cls = int(box.cls[0])
        name = results[0].names[cls]

        if name == "backpack":
            threading.Thread(target=backpack_sounds).start()  # Open textbox in a new thread
        
        if name not in ["bottle", "cell phone","potted plant","cup","backpack"]:
            continue
            
        # Get coordinates and confidence
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = float(box.conf[0])
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw the box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with confidence
        label = f"{name} {confidence:.2f}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return annotated_frame

# YOLO model
yolo_model = YOLO('object_models/yolo11s.pt')

# Open video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

run_model = False

def listen_for_input():
    global run_model
    while True:
        user_input = input()
        if user_input.lower() == 'obj':
            run_model = True
            print("MODELO de OBJETOS A CORRER")
        elif user_input.lower() == 'stop':
            run_model = False
            print("MODELO de OBJETOS PAUSADO")

# Start a thread for listening for inputs
input_thread = threading.Thread(target=listen_for_input)
input_thread.start()

screen_width = 900 
screen_height = 540 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the frame to fit the screen size
    frame = cv2.resize(frame, (screen_width, screen_height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape

    if run_model:
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        results = yolo_model(frame)
        
        # Update current instrument based on detected objects
        detected_bottle = False
        detected_phone = False
        detected_plant = False
        detected_cup = False
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                name = r.names[cls]
                if name == "bottle":
                    detected_bottle = True
                elif name == "cell phone":
                    detected_phone = True
                elif name == "potted plant":
                    detected_plant = True
                elif name == "cup":
                    detected_cup = True
                    

        # Switch instrument based on detection
        if detected_bottle:
            current_instrument = "bongo"
        elif detected_plant:
            current_instrument = "drums"
        elif detected_phone:
            current_instrument = "piano"
        elif detected_cup:
            play_random_music()

        # Draw only bottles and phones
        annotated_frame = draw_detections(frame, results)
    else:
        annotated_frame = frame

    # [Rest of the code remains the same until the hand gesture detection part...]
    # Hand detection
    hand_results = hands.process(frame_rgb)
    data_aux = []
    x_ = []
    y_ = []

    if hand_results.multi_hand_landmarks:
        # Check for two hands
        if len(hand_results.multi_hand_landmarks) == 2:
            if not two_hands_previously_shown:
                toggle_metronome()
                two_hands_previously_shown = True

        else:
            two_hands_previously_shown = False

        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Coordinates for the question mark (you have this part already in the code)
            question_mark_x = 840  # X position of the question mark
            question_mark_y = 30   # Y position of the question mark
            question_mark_width = 40  # Width of the "?" text
            question_mark_height = 60  # Height of the "?" text

            # Add a condition to check if the index finger is hovering over the question mark
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_tip_x = int(index_finger_tip.x * W)
            index_tip_y = int(index_finger_tip.y * H)

            hover_image = cv2.imread('./img_utils/help.png')  # Replace with your image path
            hover_image = cv2.resize(hover_image, (700, 500))  # Resize as needed

            # Check if the index finger tip is near the question mark
            if question_mark_x <= index_tip_x <= question_mark_x + question_mark_width and \
            question_mark_y <= index_tip_y <= question_mark_y + question_mark_height:
                cv2.imshow('Hover Image', hover_image)
            # Print the coordinates of the index finger tip
            #print(f"Index Finger Tip Coordinates: X={index_tip_x}, Y={index_tip_y}")

            # Add optional visualization on the frame
            cv2.circle(frame, (index_tip_x, index_tip_y), 10, (255, 0, 0), -1)

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
            tocar_instrumento(predicted_character, head_tilt_flag)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(annotated_frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        except Exception as e:
            pass

    # Face mesh detection
    face_results = face_mesh.process(frame_rgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[152]
            forehead = face_landmarks.landmark[10]

            image_height, image_width, _ = frame.shape
            nose_y = nose_tip.y * image_height
            chin_y = chin.y * image_height
            forehead_y = forehead.y * image_height

            face_height = abs(forehead_y - chin_y)
            if face_height == 0:
                continue

            normalized_tilt = (chin_y - nose_y) / face_height

            if normalized_tilt > 0.5:
                head_tilt_flag = "cima"
            elif normalized_tilt < 0.4:
                head_tilt_flag = "baixo"
            else:
                head_tilt_flag = "neutral"

            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            cv2.putText(annotated_frame, f"Inclinacao da cabeca: {head_tilt_flag}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, "?", (840, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)


    # Display current instrument
    cv2.putText(annotated_frame, f"Instrumento: {current_instrument}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame with updated contents
    cv2.imshow('Window 1 - Hand and Face Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        metronome_active = False
        if metronome_thread:
            metronome_thread.join()
            
blender_sender.cleanup()
cap.release()
cv2.destroyAllWindows()
