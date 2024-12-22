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
LEFT_CLICK = 0
ACTIVE_COLOR = RED

def separateWorkSpace(frame):
    h, w, c = frame.shape
    middle_start = int(h * (1 - WORKSPACE) / 2)
    middle_end = int(h * (1 + WORKSPACE) / 2)
    
    # Draw lines to separate the top, middle, and bottom sections
    cv2.line(frame, (0, middle_start), (w, middle_start), BLACK, 2)
    cv2.line(frame, (0, middle_end), (w, middle_end), BLACK, 2)
    
    return frame

def getFingerPosition(handLandmark, tipIndex):
    pos = []
    for i in range(4):
        t = handLandmark.landmark[tipIndex-i]
        pos.append((t.x, t.y, t.z))
    return pos



def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def is_finger_straight(joints):
    """
    Check if a finger is straight.
    :param joints: List of 4 joints [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
    :return: True if the finger is straight, False otherwise
    """
    # Calculate slopes between consecutive joints
    print(joints,type(joints))
    slopes = []
    for i in range(len(joints) - 1):
        dx = joints[i+1][0] - joints[i][0]
        dy = joints[i+1][1] - joints[i][1]
        # dz = joints[i+1][2] - joints[i][2]
        magnitude = math.sqrt(dx**2 + dy**2 )
        if magnitude == 0:  # Prevent division by zero
            return False
        slopes.append((dx / magnitude, dy / magnitude,))
    
    # Check if slopes are similar (fingers are straight)
    for i in range(len(slopes) - 1):
        if not math.isclose(slopes[i][0], slopes[i+1][0], rel_tol=0.1) or \
           not math.isclose(slopes[i][1], slopes[i+1][1], rel_tol=0.1) :
        #    not math.isclose(slopes[i][2], slopes[i+1][2], rel_tol=0.1):
            return False
    return True

def is_click(wrist, index_finger_joints, middle_finger_joints):
    """
    Determine if a click gesture is performed.
    :param wrist: Position of the wrist (x, y, z)
    :param index_finger_joints: Positions of index finger joints [(x1, y1, z1), ...]
    :param middle_finger_joints: Positions of middle finger joints [(x1, y1, z1), ...]
    :return: True if click gesture is detected, False otherwise
    """
    # Check if both fingers are straight
    if not is_finger_straight(index_finger_joints) or not is_finger_straight(middle_finger_joints):
        print("Fingers are not straight")
        return False

    # Check if fingers are joined (distances between corresponding joints are small)
    for i in range(len(index_finger_joints)):
        if euclidean_distance(index_finger_joints[i], middle_finger_joints[i]) > 0.02:  # Threshold for "joined"
            print("Fingers are not joined")
            return False

    # Additional check: Ensure the fingers are extended forward from the wrist
    index_tip = index_finger_joints[-1]
    if euclidean_distance(wrist, index_tip) < 0.05:  # Ensure index finger is extended enough
        print("Index finger is not extended")
        return False

    return True



def pressOption(handLandmark):
    global LEFT_CLICK

    # Get the coordinates of the index finger tip (Landmark 8)
    t = handLandmark.landmark[mp_hands.HandLandmark.WRIST]
    wrist = (t.x, t.y, t.z)
    indTip, indDip, indPip, indMcp = getFingerPosition(handLandmark, mp_hands.HandLandmark.INDEX_FINGER_TIP)
    midTip, midDip, midPip, midMcp = getFingerPosition(handLandmark, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
    


    # use above variables to determine if the user is clicking
    if is_click(wrist, [indTip, indDip, indPip, indMcp], [midTip, midDip, midPip, midMcp]):
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
                pressOption(hand_landmarks)
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
    frame = separateWorkSpace(frame)
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