import cv2
import mediapipe as mp
import math

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

# state variables
INDEX_ACTIVE = 0
MIDDLE_ACTIVE = 0
LEFT_CLICK = 0
ACTIVE_COLOR = RED

def getFingerPosition(handLandmark, tipIndex):
    pos = []
    for i in range(4):
        t = handLandmark.landmark[tipIndex-i]
        pos.append((t.x, t.y, t.z))
    return pos

def angle_between_vectors(v1, v2):
    """Calculate the angle (in degrees) between two vectors."""
    dot_product = sum(v1[i] * v2[i] for i in range(3))
    magnitude_v1 = math.sqrt(sum(v1[i] ** 2 for i in range(3)))
    magnitude_v2 = math.sqrt(sum(v2[i] ** 2 for i in range(3)))
    if magnitude_v1 == 0 or magnitude_v2 == 0:  # Avoid division by zero
        return 0
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    cos_theta = max(-1, min(1, cos_theta))  # Clamp to avoid numerical errors
    return math.degrees(math.acos(cos_theta))

def is_finger_straight(wrist, joints):
    """
    Check if a finger is straight.
    :param joints: List of 4 joints [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
    :return: True if the finger is straight, False otherwise
    """
    # Calculate slopes between consecutive joints
    slopes = []
    for i in range(len(joints) - 1):
        dx = joints[i+1][0] - joints[i][0]
        dy = joints[i+1][1] - joints[i][1]
        dz = joints[i+1][2] - joints[i][2]
        magnitude = math.sqrt(dx**2 + dy**2 + dz**2)
        if magnitude == 0:  # Prevent division by zero
            return False
        slopes.append((dx / magnitude, dy / magnitude, dz / magnitude))

    # Calculate the angle between consecutive slopes
    angles = []
    for i in range(len(slopes) - 1):
        angle = angle_between_vectors(slopes[i], slopes[i+1])
        angles.append(angle)

    # Check if the angles are within a threshold
    return all(angle < 30 for angle in angles)

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def is_left_click(wrist, index_finger_joints, middle_finger_joints):
    """
    Check if a left click is performed.
    :param wrist: Tuple containing the wrist position (x, y, z)
    :param index_finger_joints: List of 4 joints of the index finger [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
    :param middle_finger_joints: List of 4 joints of the middle finger [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
    :return: True if a left click is performed, False otherwise
    """
    # Assumed that index and middle fingers are straight
    """
    # # Check if the index finger is straight
    # if not is_finger_straight(wrist, index_finger_joints):
    #     return False

    # # Check if the middle finger is straight
    # if not is_finger_straight(wrist, middle_finger_joints):
    #     return False
    """
    
    # Check if the distance between the index and middle finger tips is small
    avg_dist = (euclidean_distance(index_finger_joints[0], middle_finger_joints[0]) +
                euclidean_distance(index_finger_joints[1], middle_finger_joints[1]) +
                euclidean_distance(index_finger_joints[2], middle_finger_joints[2])) / 3
    return avg_dist < 0.05

def eventHandler(hand_landmarks):
    global LEFT_CLICK
    
    t = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    wrist = (t.x, t.y, t.z)
    index_finger_joints = getFingerPosition(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP)
    middle_finger_joints = getFingerPosition(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)


    # Check if the click gesture is performed
    if is_finger_straight(wrist, index_finger_joints):
        INDEX_ACTIVE = 1
    else:
        INDEX_ACTIVE = 0
    
    if is_finger_straight(wrist, middle_finger_joints):
        MIDDLE_ACTIVE = 1
    else:
        MIDDLE_ACTIVE = 0

    if INDEX_ACTIVE == 1 and MIDDLE_ACTIVE == 1 and is_left_click(wrist, index_finger_joints, middle_finger_joints):
        LEFT_CLICK = 1
    else:
        LEFT_CLICK = 0


def processHand(frame):
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
                # pressOption(hand_landmarks)
                eventHandler(hand_landmarks)
                if(id == 8):
                    cv2.circle(frame, (cx, cy), 15,GREEN, cv2.FILLED)
                # print(f"ID: {id}, Position: ({cx}, {cy})")
    return frame

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while cap.isOpened():
    # Capture frame-by-frame
    LEFT_CLICK = 0
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame = processHand(frame)
    # frame = separateWorkSpace(frame)
    if LEFT_CLICK == 1:
       cv2.putText(frame, "LEFT CLICK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()