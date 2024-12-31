import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start webcam feed
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize head tilt flag
head_tilt_flag = "neutral"

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Perform face mesh detection
        results = face_mesh.process(frame_rgb)

        # Convert back to BGR for OpenCV
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Process detected landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get key landmarks for tilt detection
                nose_tip = face_landmarks.landmark[1]  # Nose tip
                chin = face_landmarks.landmark[152]   # Chin
                forehead = face_landmarks.landmark[10]  # Forehead (used for bounding box height)

                # Get screen space coordinates
                image_height, image_width, _ = frame_bgr.shape
                nose_y = nose_tip.y * image_height
                chin_y = chin.y * image_height
                forehead_y = forehead.y * image_height

                # Calculate the face height as a reference length
                face_height = abs(forehead_y - chin_y)

                # Ensure face_height is non-zero to avoid division errors
                if face_height == 0:
                    print("Error: Face height is zero. Skipping frame.")
                    continue

                # Normalize tilt based on face height
                normalized_tilt = (chin_y - nose_y) / face_height

                # Debugging output
                print(f"Nose Y: {nose_y}, Chin Y: {chin_y}, Forehead Y: {forehead_y}, Face Height: {face_height}")
                print(f"Normalized Tilt: {normalized_tilt:.2f}")

                # Update head tilt flag based on normalized tilt value
                if normalized_tilt > 0.5:  # Adjust thresholds as needed
                    head_tilt_flag = "upwards"
                elif normalized_tilt < 0.4:  # Adjust thresholds as needed
                    head_tilt_flag = "downwards"
                else:
                    head_tilt_flag = "neutral"

                # Print the flag
                print(f"Head Tilt Flag: {head_tilt_flag}")

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        # Show frame
        cv2.imshow('Head Tilt Detection', frame_bgr)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()