import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection and Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start webcam feed
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or specify another index for external cameras

with mp_face_mesh.FaceMesh(
    static_image_mode=False,     # For live video, set to False
    max_num_faces=1,             # Detect a single face; increase for multi-face detection
    refine_landmarks=True,       # Enables iris landmarks
    min_detection_confidence=0.5,  # Minimum confidence for detection
    min_tracking_confidence=0.5   # Minimum confidence for tracking
) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Convert the BGR image to RGB (MediaPipe works with RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # Improve performance

        # Perform face mesh detection
        results = face_mesh.process(frame_rgb)

        # Convert back to BGR for OpenCV operations
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw face mesh landmarks if detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        # Display the processed frame
        cv2.imshow('MediaPipe Face Mesh', frame_bgr)

        # Exit loop on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
