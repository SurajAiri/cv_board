import math
import mediapipe as mp
mp_hands = mp.solutions.hands  

class HandEventHandler:
    def __init__(self):
        self.left_active = False
        self.middle_active = False
        self.left_pressed = False

    def get_finger_position(self, hand_landmark, tip_index):
        pos = []
        for i in range(4):
            t = hand_landmark.landmark[tip_index - i]
            pos.append((t.x, t.y, t.z))
        return pos

    def angle_between_vectors(self, v1, v2):
        """Calculate the angle (in degrees) between two vectors."""
        dot_product = sum(v1[i] * v2[i] for i in range(3))
        magnitude_v1 = math.sqrt(sum(v1[i] ** 2 for i in range(3)))
        magnitude_v2 = math.sqrt(sum(v2[i] ** 2 for i in range(3)))
        if magnitude_v1 == 0 or magnitude_v2 == 0:  # Avoid division by zero
            return 0
        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
        cos_theta = max(-1, min(1, cos_theta))  # Clamp to avoid numerical errors
        return math.degrees(math.acos(cos_theta))

    def is_finger_straight(self, wrist, joints):
        """
        Check if a finger is straight.
        :param joints: List of 4 joints [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
        :return: True if the finger is straight, False otherwise
        """
        # Calculate slopes between consecutive joints
        slopes = []
        for i in range(len(joints) - 1):
            dx = joints[i + 1][0] - joints[i][0]
            dy = joints[i + 1][1] - joints[i][1]
            dz = joints[i + 1][2] - joints[i][2]
            magnitude = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if magnitude == 0:  # Prevent division by zero
                return False
            slopes.append((dx / magnitude, dy / magnitude, dz / magnitude))

        # Calculate the angle between consecutive slopes
        angles = []
        for i in range(len(slopes) - 1):
            angle = self.angle_between_vectors(slopes[i], slopes[i + 1])
            angles.append(angle)

        # Check if the angles are within a threshold
        return all(angle < 30 for angle in angles)

    def euclidean_distance(self, p1, p2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def __is_left_click__(self, wrist, index_finger_joints, middle_finger_joints):
        """
        Check if a left click is performed.
        :param wrist: Tuple containing the wrist position (x, y, z)
        :param index_finger_joints: List of 4 joints of the index finger [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
        :param middle_finger_joints: List of 4 joints of the middle finger [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
        :return: True if a left click is performed, False otherwise
        """
        # Check if the distance between the index and middle finger tips is small
        avg_dist = (self.euclidean_distance(index_finger_joints[0], middle_finger_joints[0]) +
                    self.euclidean_distance(index_finger_joints[1], middle_finger_joints[1]) +
                    self.euclidean_distance(index_finger_joints[2], middle_finger_joints[2])) / 3
        return avg_dist < 0.05

    def event_handler(self, hand_landmarks):
        t = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        wrist = (t.x, t.y, t.z)
        index_finger_joints = self.get_finger_position(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP)
        middle_finger_joints = self.get_finger_position(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)

        # Check if the click gesture is performed
        self.left_active = self.is_finger_straight(wrist, index_finger_joints)
        self.middle_active = self.is_finger_straight(wrist, middle_finger_joints)

        if self.left_active and self.middle_active and self.__is_left_click__(wrist, index_finger_joints, middle_finger_joints):
            self.left_pressed = True
        else:
            self.left_pressed = False

        # Update position of index finger tip x and y
        self.index_tip = index_finger_joints[0][:2]
