import subprocess

try:
    # Install opencv-python using pip
    subprocess.run(["pip", "install", "opencv-python"])
    print("OpenCV installed successfully!")
except Exception as e:
    print(f"Error installing OpenCV: {e}")

# All the imports go here
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

class AirCanvas:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        # Added attributes for window size
        self.window_width = 800
        self.window_height = 600

        # Initialize ret
        self.ret = True
        # Added attributes for canvas size
        self.canvas_width = 800
        self.canvas_height = 600

        # Initialize paintWindow
        self.paintWindow = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) + 255

        # Added attributes for rectangles, colors, and button_labels
        self.rectangles = [(40, 1, 140, 65), (160, 1, 255, 65), (275, 1, 370, 65), (390, 1, 485, 65), (505, 1, 600, 65)]
        self.colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        self.button_labels = ["CLEAR", "BLUE", "GREEN", "RED", "YELLOW"]

        # Initialize color-related attributes
        self.bpoints = [deque(maxlen=512)]
        self.gpoints = [deque(maxlen=512)]
        self.rpoints = [deque(maxlen=512)]
        self.ypoints = [deque(maxlen=512)]
        self.wpoints = [deque(maxlen=512)]

        self.blue_index = 0
        self.green_index = 0
        self.red_index = 0
        self.yellow_index = 0
        self.white_index = 0

        self.colorIndex = 0

        # Initialize other attributes
        self.kernel = np.ones((5, 5), np.uint8)

        # Here is code for Canvas setup
        self.paintWindow = np.zeros((600,800,3), dtype=np.uint8) + 255
        rectangles = [(40, 1, 140, 65), (160, 1, 255, 65), (275, 1, 370, 65), (390, 1, 485, 65), (505, 1, 600, 65)]
        button_labels = ["CLEAR", "BLUE", "GREEN", "RED", "YELLOW"]

        for rect, label in zip(rectangles, button_labels):
            self.paintWindow = cv2.rectangle(self.paintWindow, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), 2)
            cv2.putText(self.paintWindow, label, (rect[0] + 9, rect[1] + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                        cv2.LINE_AA)

        cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

        # Initialize mediapipe
        mpHands = mp.solutions.hands
        self.hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils

    def setup_canvas(self):
        for i, (rect, color, label) in enumerate(zip(self.rectangles, self.colors, self.button_labels)):
            self.paintWindow = cv2.rectangle(self.paintWindow, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
            cv2.putText(self.paintWindow, label, (rect[0] + 9, rect[1] + 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    def handle_buttons(self, center):
        for i, rect in enumerate(self.rectangles):
            if rect[0] <= center[0] <= rect[2]:
                if i == 0:
                    self.bpoints = [deque(maxlen=512)]
                    self.gpoints = [deque(maxlen=512)]
                    self.rpoints = [deque(maxlen=512)]
                    self.ypoints = [deque(maxlen=512)]
                    self.wpoints = [deque(maxlen=512)]

                    self.blue_index = 0
                    self.green_index = 0
                    self.red_index = 0
                    self.yellow_index = 0
                    self.white_index = 0

                    self.paintWindow[67:, :, :] = 255
                else:
                    self.colorIndex = i - 1

    def draw_lines(self, points):
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(self.frame, points[i][j][k - 1], points[i][j][k], self.colors[i][:3], 2)
                    cv2.line(self.paintWindow, points[i][j][k - 1], points[i][j][k], self.colors[i][:3], 2)

    def run(self):
        while self.ret:
            self.ret, self.frame = self.cap.read()
            self.frame = cv2.flip(self.frame, 1)
            self.framergb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            # Resize canvas
            enlarged_canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) + 255
            enlarged_canvas = cv2.resize(enlarged_canvas, (self.frame.shape[1], self.frame.shape[0]))
            combined_frame = cv2.addWeighted(self.frame, 1, enlarged_canvas, 0.5, 0)
            # Resize the windows
            cv2.setWindowProperty('Paint', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Paint', self.window_width, self.window_height)

            # Setup color palette
            self.setup_canvas()

            result = self.hands.process(self.framergb)

            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * 640)
                        lmy = int(lm.y * 480)
                        landmarks.append([lmx, lmy])

                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(self.frame, handslms, mpHands.HAND_CONNECTIONS)
                fore_finger = (landmarks[8][0], landmarks[8][1])
                center = fore_finger
                thumb = (landmarks[4][0], landmarks[4][1])
                cv2.circle(self.frame, center, 3, (0, 255, 0), -1)

                if thumb[1] - center[1] < 30:
                    # Append new deques when thumb is close to the forefinger
                    self.bpoints.append(deque(maxlen=512))
                    self.blue_index += 1
                    self.gpoints.append(deque(maxlen=512))
                    self.green_index += 1
                    self.rpoints.append(deque(maxlen=512))
                    self.red_index += 1
                    self.ypoints.append(deque(maxlen=512))
                    self.yellow_index += 1

                elif center[1] <= 65:
                    # Button logic
                    for i, rect in enumerate(self.rectangles):
                        if rect[0] <= center[0] <= rect[2]:
                            if i == 0:
                                # Clear Button
                                self.bpoints = [deque(maxlen=512)]
                                self.gpoints = [deque(maxlen=512)]
                                self.rpoints = [deque(maxlen=512)]
                                self.ypoints = [deque(maxlen=512)]

                                self.blue_index = 0
                                self.green_index = 0
                                self.red_index = 0
                                self.yellow_index = 0

                                self.paintWindow[67:, :, :] = 255
                            else:
                                self.colorIndex = i - 1
                else:
                    # Drawing logic
                    if self.colorIndex == 0:
                        self.bpoints[self.blue_index].appendleft(center)
                    elif self.colorIndex == 1:
                        self.gpoints[self.green_index].appendleft(center)
                    elif self.colorIndex == 2:
                        self.rpoints[self.red_index].appendleft(center)
                    elif self.colorIndex == 3:
                        self.ypoints[self.yellow_index].appendleft(center)

            else:
                # Append new deques when nothing is detected to avoid messing up
                self.bpoints.append(deque(maxlen=512))
                self.blue_index += 1
                self.gpoints.append(deque(maxlen=512))
                self.green_index += 1
                self.rpoints.append(deque(maxlen=512))
                self.red_index += 1
                self.ypoints.append(deque(maxlen=512))
                self.yellow_index += 1

            # Draw lines of all the colors on the canvas and frame
            points = [self.bpoints, self.gpoints, self.rpoints, self.ypoints, self.wpoints]
            self.draw_lines(points)

            # Display the frame
            combined_frame = cv2.addWeighted(self.frame, 1, self.paintWindow, 0.5, 0)
            cv2.imshow("Output", combined_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    air_canvas = AirCanvas()
    air_canvas.run()
