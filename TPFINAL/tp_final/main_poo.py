import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time
from ultralytics import YOLO

class SoundPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {
            'a': pygame.mixer.Sound('./piano/do-stretched.wav'),
            'b': pygame.mixer.Sound('./piano/re-stretched.wav'),
            'c': pygame.mixer.Sound('./piano/mi-stretched.wav'),
            'd': pygame.mixer.Sound('./piano/fa-stretched.wav'),
            'i': pygame.mixer.Sound('./piano/sol-stretched.wav')
        }
        self.metronome_sound = pygame.mixer.Sound('./metronome/click.wav')
        self.sound_played = {key: False for key in self.sounds}
        self.metronome_active = False

    def play_sound(self, key):
        if key in self.sounds and not self.sound_played[key]:
            self.sounds[key].play()
            self.sound_played[key] = True

    def reset_sounds(self):
        for key in self.sound_played:
            self.sound_played[key] = False

    def play_metronome(self):
        self.metronome_active = not self.metronome_active
        if self.metronome_active:
            threading.Thread(target=self._metronome_loop).start()

    def _metronome_loop(self):
        while self.metronome_active:
            self.metronome_sound.play()
            time.sleep(0.6)  # Assuming 100 BPM

class ObjectDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        return self.model(frame)

class GestureRecognition:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    def detect_gestures(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)

class FaceMeshDetection:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def detect_face(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(frame_rgb)

class MainApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.sound_player = SoundPlayer()
        self.object_detection = ObjectDetection('object_models/yolo11s.pt')
        self.gesture_recognition = GestureRecognition()
        self.face_mesh_detection = FaceMeshDetection()
        self.run_object_detection = False

    def toggle_object_detection(self):
        self.run_object_detection = not self.run_object_detection

    def process_frame(self, frame):
        if self.run_object_detection:
            detections = self.object_detection.detect_objects(frame)
            for det in detections:
                # Draw bounding boxes and labels
                # Further processing...
                pass

        gestures = self.gesture_recognition.detect_gestures(frame)
        if gestures.multi_hand_landmarks:
            for hand_landmarks in gestures.multi_hand_landmarks:
                # Process hand landmarks
                # Example: self.sound_player.play_sound('a')
                pass

        face = self.face_mesh_detection.detect_face(frame)
        if face.multi_face_landmarks:
            for face_landmarks in face.multi_face_landmarks:
                # Process face landmarks
                pass

        return frame

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('Hand and Face Detection', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.sound_player.play_metronome()
            elif key == ord('o'):
                self.toggle_object_detection()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MainApp()
    app.run()
