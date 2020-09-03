import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import dlib
import ctypes


# Initializing the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.',
                    default='venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')

args = parser.parse_args()
face_cascade_name = args.face_cascade
face_cascade = cv2.CascadeClassifier()

# -- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

cap = cv2.VideoCapture("testDriver.mp4")

fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
out = cv2.VideoWriter('output.mp4', fourcc, 29, (1080, 1920))
font = cv2.FONT_HERSHEY_SIMPLEX

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
print("SCREEN_SIZE: ", screensize)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

print('--(*)Success in opening video capture')


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


# Defining the mid-point
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


# Defining the Euclidean distance
def euclidean_distance(leftx, lefty, rightx, righty):
    return np.sqrt((leftx - rightx) ** 2 + (lefty - righty) ** 2)


# Defining the eye aspect ratio
def get_EAR(eye_points, facial_landmarks):
    # Defining the left point of the eye
    left_point = [facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y]
    # Defining the right point of the eye
    right_point = [facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y]
    # Defining the top mid-point of the eye
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    # Defining the bottom mid-point of the eye
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    # Drawing horizontal and vertical line
    hor_line = cv2.line(frame, (left_point[0], left_point[1]), (right_point[0], right_point[1]), (255, 0, 0), 3)
    ver_line = cv2.line(frame, (center_top[0], center_top[1]), (center_bottom[0], center_bottom[1]), (255, 0, 0), 3)
    # Calculating length of the horizontal and vertical line
    hor_line_lenght = euclidean_distance(left_point[0], left_point[1], right_point[0], right_point[1])
    ver_line_lenght = euclidean_distance(center_top[0], center_top[1], center_bottom[0], center_bottom[1])
    # Calculating eye aspect ratio
    EAR = ver_line_lenght / hor_line_lenght
    return EAR


# Creating a list eye_blink_signal
eye_blink_signal = []
# Creating an object blink_ counter
blink_counter = 0
previous_ratio = 100
# Creating a while loop

fps = 59.95
duration = 64

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converting a color frame into a grayscale frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Creating an object in which we will sore detected faces
    # faces = detector(gray)

    blinking_ratio_rounded = 0
    frame_gray = cv2.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        # Creating an object in which we will sore detected facial landmarks
        landmarks = predictor(frame_gray, face)
        # Calculating left eye aspect ratio
        left_eye_ratio = get_EAR([36, 37, 38, 39, 40, 41], landmarks)
        # Calculating right eye aspect ratio
        right_eye_ratio = get_EAR([42, 43, 44, 45, 46, 47], landmarks)
        # Calculating aspect ratio for both eyes
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        # Rounding blinking_ratio on two decimal places
        blinking_ratio_1 = blinking_ratio * 100
        blinking_ratio_2 = np.round(blinking_ratio_1)
        blinking_ratio_rounded = blinking_ratio_2 / 100
        # Appending blinking ratio to a list eye_blink_signal
        eye_blink_signal.append(blinking_ratio)
        if blinking_ratio < 0.20:
            if previous_ratio > 0.20:
                blink_counter = blink_counter + 1
        # Displaying blink counter and blinking ratio in our output video

        previous_ratio = blinking_ratio

    frame = cv2.putText(frame, str(blink_counter), (30, 50), font, 2, (0, 0, 255), 5)
    frame = cv2.putText(frame, str(blinking_ratio_rounded), (900, 50), font, 2, (0, 0, 255), 5)
    cv2.imshow('Capture - Face detection', image_resize(frame, height=600))
    if cv2.waitKey(10) == 27:
        break
    out.write(frame)
out.release()
