import time
import cv2
import mediapipe as mp
import math
from event_handler import HandEventHandler
import numpy as np

from utils import DrawOption

# # Open a connection to the camera
cap = cv2.VideoCapture(1)
mp_hands = mp.solutions.hands  
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7,max_num_hands=1)

#COLORS
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
ORANGE = (0, 165, 255)
PINK = (147, 20, 255)
CYAN = (255, 255, 0)
GREY = (128, 128, 128)
MAGENTA = (255, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def separateWorkSpace(w,h):
    middle_start = int(h * (1 - WORKSPACE) / 2)
    middle_end = int(h * (1 + WORKSPACE) / 2)
    
    return ((0, middle_start), (w, middle_start),(0, middle_end), (w, middle_end))

WORKSPACE = 0.8
WIDTH, HEIGHT = 1920, 1080
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

WORKSPACE_AREA = separateWorkSpace(WIDTH,HEIGHT)


CANVAS = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
TEMP_CANVAS = np.zeros((HEIGHT, WIDTH, 3), np.uint8)


    
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

# def onLeftClick():
#     # print("Left Click")

state = HandEventHandler()

class CVButton:
    BTN_SIZE = (200, 80)
    def __init__(self, origin, graphic,bgColor,action,label):
        self.origin = origin
        self.graphic = graphic
        self.bg = bgColor
        self.action = action
        self.label = label

    def isClicked(self, pos):
        x, y = pos
        x1, y1 = self.origin
        x2, y2 = (x1 + self.BTN_SIZE[0], y1 + self.BTN_SIZE[1])
        return x1 <= x <= x2 and y1 <= y <= y2


# btnOrigin, graphic
btn_pressed_time = time.time()
BUTTONS = []


# Button actions
def action_clear_canvas():
    global CANVAS
    CANVAS = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    state.draw_option = DrawOption.NONE
    print("Canvas cleared")

def set_draw_option(option):
    state.draw_option = option

def set_color(color):
    state.active_color = color

# Initialize buttons
BUTTONS = [
    CVButton((20, 20), None, WHITE, lambda: set_draw_option(DrawOption.PEN), "Pen"),
    CVButton((240, 20), None, WHITE, lambda: set_draw_option(DrawOption.LINE), "Line"),
    CVButton((460, 20), None, WHITE, lambda: set_draw_option(DrawOption.TRIANGLE), "Triangle"),
    CVButton((680, 20), None, WHITE, lambda: set_draw_option(DrawOption.CIRCLE), "Circle"),
    CVButton((900, 20), None, WHITE, lambda: set_draw_option(DrawOption.RECTANGLE), "Rectangle"),
    CVButton((1120, 20), None, WHITE, lambda: set_draw_option(DrawOption.ERASE), "Eraser"),
    CVButton((1340, 20), None, WHITE, action_clear_canvas, "Clear"),
]

STATIC_BUTTONS = [
    CVButton((1560, 20), None, state.active_color, lambda: None, ""),
    # bottom buttons for colors
    CVButton((20, 1000), None, RED, lambda: set_color(RED), "Red"),
    CVButton((240, 1000), None, GREEN, lambda: set_color(GREEN), "Green"),
    CVButton((460, 1000), None, BLUE, lambda: set_color(BLUE), "Blue"),
    CVButton((680, 1000), None, ORANGE, lambda: set_color(ORANGE), "Orange"),
    CVButton((900, 1000), None, PINK, lambda: set_color(PINK), "Pink"),
    CVButton((1120, 1000), None, CYAN, lambda: set_color(CYAN), "Cyan"),
    CVButton((1340, 1000), None, GREY, lambda: set_color(GREY), "Grey"),
    CVButton((1560, 1000), None, MAGENTA, lambda: set_color(MAGENTA), "Magenta")
]


# Draw buttons on canvas
def draw_buttons(canvas):
    for button in BUTTONS:
        x, y = button.origin
        color = button.bg
        cv2.rectangle(canvas, (x, y), (x + button.BTN_SIZE[0], y + button.BTN_SIZE[1]), color, -1)
        cv2.putText(canvas, button.label, (x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)

    for button in STATIC_BUTTONS:
        x, y = button.origin
        color = button.bg
        cv2.rectangle(canvas, (x, y), (x + button.BTN_SIZE[0], y + button.BTN_SIZE[1]), color, -1)
        cv2.putText(canvas, button.label, (x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)


def uiButtonHandler(canvas):
    global button_pressed_time,first_point
    draw_buttons(canvas)
    if not state.left_pressed:
        return
    
    if time.time() - button_pressed_time < 0.3:
        return
    
    pressedInd = -1
    for i in range(len(BUTTONS)):
        if BUTTONS[i].isClicked(state.pos):
            print("Button clicked")
            button_pressed_time = time.time()
            BUTTONS[i].action()
            first_point = None
            pressedInd = i
            break
    
    for btn in STATIC_BUTTONS:
        if btn.isClicked(state.pos):
            button_pressed_time = time.time()
            btn.action()
            STATIC_BUTTONS[0].bg = state.active_color
            first_point = None


    if pressedInd != -1:
        for i in range(len(BUTTONS)):
            if i == pressedInd:
                BUTTONS[i].bg = GREEN
            else:
                BUTTONS[i].bg = WHITE



first_point = None
button_pressed_time = time.time()

def draw(canvas, pt1, pt2, opt:DrawOption):
    if opt == DrawOption.LINE:
        cv2.line(canvas, pt1, pt2, state.active_color, 5)
    elif opt == DrawOption.RECTANGLE:
        cv2.rectangle(canvas, pt1, pt2, state.active_color, 5)
    elif opt == DrawOption.CIRCLE:
        cv2.circle(canvas, pt1, int(math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)), state.active_color, 5) 
    elif opt == DrawOption.TRIANGLE:
        cv2.line(canvas, pt1, pt2, state.active_color, 5)
        cv2.line(canvas, pt2, (pt1[0], pt2[1]), state.active_color, 5)
        cv2.line(canvas, (pt1[0], pt2[1]), pt1, state.active_color, 5)

def highlight_first_point(canvas, pt):
    cv2.circle(canvas, pt, 5, GREEN, cv2.FILLED)

def drawShapes():
    global  CANVAS, TEMP_CANVAS, first_point, button_pressed_time
    # print("Drawing shapes",first_point,state.left_pressed)
    delta = time.time() - button_pressed_time
    
    if first_point is not None:
        highlight_first_point(TEMP_CANVAS, first_point)

    if delta < 0.6:
        return
    
    if first_point is None: 
        if state.left_pressed:
            print("pressed",button_pressed_time)
            button_pressed_time = time.time()
            first_point = state.pos
            return
        else:
            return
    
    # print(time.time() - button_pressed_time)
    # ignore if pos is outside the workspace
    if state.pos[1] < WORKSPACE_AREA[0][1] or state.pos[1] > WORKSPACE_AREA[2][1]:
        return
    
    if state.left_pressed:
        print("Drawing",button_pressed_time)
        # actual drawing
        draw(CANVAS, first_point, state.pos, state.draw_option)
        first_point = None
        button_pressed_time = time.time()
    else:
        # print("Not Drawing")
        draw(TEMP_CANVAS, first_point, state.pos, state.draw_option)


prev_pos = (0, 0)
def drawHandler():
    global CANVAS,TEMP_CANVAS, prev_pos
    TEMP_CANVAS = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    if state.draw_option == DrawOption.NONE or not state.index_active :
        return
    # print(state.draw_option)
    if state.draw_option == DrawOption.PEN or state.draw_option == DrawOption.ERASE:
        # if delta is too large, it will draw a line from the previous position to the current position skip
        if (abs(state.pos[0] - prev_pos[0]) > 50) or (abs(state.pos[1] - prev_pos[1]) > 50):
            prev_pos = state.pos
            return
        col = state.active_color if state.draw_option == DrawOption.PEN else BLACK
        cv2.line(CANVAS, (prev_pos[0], prev_pos[1]), (state.pos[0], state.pos[1]), col,5)
        prev_pos = state.pos
    else:
        drawShapes()
    
    

while cap.isOpened():
    # Capture frame-by-frame
    LEFT_CLICK = 0
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame = processHand(frame,state.event_handler)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    """if key == ord('c'):
        print("clear")
        CANVAS = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        prev_pos = (0, 0)
        state.draw_option = DrawOption.NONE
    
    if key == ord('p'):
        print("pen")
        state.draw_option = DrawOption.PEN
        state.active_color = GREEN

    if key == ord('t'):
        print("triangle")
        state.draw_option = DrawOption.TRIANGLE
    if key == ord('r'):
        print("rectangle")
        state.draw_option = DrawOption.RECTANGLE
    if key == ord('l'):
        print("line")
        state.draw_option = DrawOption.LINE
    
    if key == ord('e'):
        print("eraser")
        state.draw_option = DrawOption.ERASE
"""
    
    drawHandler()
    uiButtonHandler(TEMP_CANVAS)
    final_canvas = cv2.addWeighted( CANVAS, 1,TEMP_CANVAS, 1, 0)
    frame = cv2.addWeighted(frame, 0.5, final_canvas, 0.5, 0)
    cv2.imshow('Camera Feed', frame)

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()