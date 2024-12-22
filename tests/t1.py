import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,  # Detect continuously
                       max_num_hands=2,         # Maximum hands to detect
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start capturing from webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Flip the frame for a mirrored effect and convert color to RGB.
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands.
    results = hands.process(rgb_frame)

    # Draw hand landmarks and capture positions.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks and connections.
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Print the coordinates of each landmark.
            for id, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                print(f"ID: {id}, Position: ({cx}, {cy})")

    # Show the frame.
    cv2.imshow('MediaPipe Hands', frame)

    # Exit when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
