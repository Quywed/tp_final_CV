import threading
import logging
from ultralytics import YOLO
import cv2

def disable_YOLO_logs():
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

#YOLO model
model = YOLO('object_models/yolo11s.pt')

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
            print("MODELO A CORRER.")
        elif user_input.lower() == 'stop':
            run_model = False
            print("MODELO PAUSADO.")

input_thread = threading.Thread(target=listen_for_input)
input_thread.start()

while True:

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break


    if run_model:
        disable_YOLO_logs() 
        results = model(frame)
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame

    cv2.imshow('FRAME', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()