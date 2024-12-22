import cv2
import mediapipe as mp
import numpy as np

class DrawingBoard:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_color = (0, 0, 255)  # Default color: red
        self.brush_thickness = 5

        # Canvas for drawing
        self.canvas = None
        self.prev_x, self.prev_y = None, None  # For freehand drawing

    def select_tool(self, key):
        tools = {
            ord('r'): (255, 0, 0),  # Red
            ord('g'): (0, 255, 0),  # Green
            ord('b'): (0, 0, 255),  # Blue
            ord('e'): (0, 0, 0)     # Eraser (Black)
        }
        self.drawing_color = tools.get(key, self.drawing_color)

    def process_frame(self, frame):
        h, w, c = frame.shape

        # Flip the frame for a mirrored effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = self.hands.process(rgb_frame)

        # Initialize canvas if not already done
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get coordinates of the index finger tip (Landmark 8)
                x_tip = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
                y_tip = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

                # Check if we have a previous point to draw a line
                if self.prev_x is not None and self.prev_y is not None:
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x_tip, y_tip), self.drawing_color, self.brush_thickness)

                self.prev_x, self.prev_y = x_tip, y_tip
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        else:
            self.prev_x, self.prev_y = None, None  # Reset when no hand is detected

        # Combine the canvas and frame
        combined_frame = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
        return combined_frame

    def run(self):
        cap = cv2.VideoCapture(1)

        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty frame.")
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow("Drawing Board - Press 'q' to Quit", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit the application
                break
            self.select_tool(key)  # Change tools based on key press

        cap.release()
        cv2.destroyAllWindows()

# Run the application
if __name__ == "__main__":
    app = DrawingBoard()
    app.run()
