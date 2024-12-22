import cv2
import mediapipe as mp
import math
from event_handler import HandEventHandler

# Open a connection to the camera
cap = cv2.VideoCapture(1)
mp_hands = mp.solutions.hands  
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7,max_num_hands=1)

#COLORS
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

WORKSPACE = 0.8


def separateWorkSpace(frame):
    h, w, c = frame.shape
    middle_start = int(h * (1 - WORKSPACE) / 2)
    middle_end = int(h * (1 + WORKSPACE) / 2)
    
    # Draw lines to separate the top, middle, and bottom sections
    cv2.line(frame, (0, middle_start), (w, middle_start), BLACK, 2)
    cv2.line(frame, (0, middle_end), (w, middle_end), BLACK, 2)
    
    return frame

    
def processHand(frame,clickHandler):
    # Flip the frame for a mirrored effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks and connections.
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Print the coordinates of each landmark.
            for id, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                clickHandler(hand_landmarks)
                if(id == 8):
                    cv2.circle(frame, (cx, cy), 15,GREEN, cv2.FILLED)
                # print(f"ID: {id}, Position: ({cx}, {cy})")
    return frame


if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ch = HandEventHandler()

while cap.isOpened():
    # Capture frame-by-frame
    LEFT_CLICK = 0
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame = processHand(frame,ch.event_handler)
    frame = separateWorkSpace(frame)
    if ch.left_pressed == 1:
       cv2.putText(frame, "LEFT CLICK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()