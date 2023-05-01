import cv2 as cv
import numpy as np
import argparse
import pyautogui
from pynput import keyboard
import time

print('Press "L" to quit')
print('Press "F" to toggle the mod')

parser = argparse.ArgumentParser()
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

def on_key_release(key):
    global tracking_enabled
    try:
        if key.char.lower() == 'f':
            tracking_enabled = not tracking_enabled
        elif key.char.lower() == 'f':
            tracking_enabled = False
    except AttributeError:
        pass

key_listener = keyboard.Listener(on_release=on_key_release)
key_listener.start()

tracking_enabled = True
while True:
    t1 = time.time()

    # Take a screenshot of the screen
    screenshot = np.array(pyautogui.screenshot())

    # Resize the screenshot to a smaller size for faster processing
    screenshot = cv.resize(screenshot, (0, 0), fx=0.5, fy=0.5)

    frameWidth = screenshot.shape[1]
    frameHeight = screenshot.shape[0]

    # Preprocess the screenshot and run it through the network
    net.setInput(cv.dnn.blobFromImage(screenshot, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body part.
        heatMap = out[0, i, :, :]

        # Find the position of the body part with the highest confidence value
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)

    # Draw lines and points to show the detected pose
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(screenshot, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(screenshot, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(screenshot, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # Get position of the head (Nose)
    head_pos = points[BODY_PARTS["Nose"]]

    if tracking_enabled and head_pos:
        pyautogui.moveTo(head_pos[0] * 2, head_pos[1] * 2)

    t2 = time.time()
    fps = 1 / (t2 - t1)
    cv.putText(screenshot, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the resulting image with the detected pose
    cv.imshow('OpenPose using OpenCV', screenshot)

    if cv.waitKey(1) == ord('l'):
        break

cv.destroyAllWindows()
key_listener.stop()