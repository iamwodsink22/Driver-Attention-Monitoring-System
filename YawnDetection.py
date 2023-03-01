import cv2
from scipy.spatial import distance as dist
import numpy as np
from audio import speak


class YawnDetect:
    def __init__(self):
        self.keypoints = None
        self.frame = None
        self.mouth_points = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
                             185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
        self.mouth = []

    def getMouthLandmark(self, frame, landmarks, width, height):
        mouth_points = []
        if landmarks:
            for land in landmarks:
                for id, lm in enumerate(land.landmark):
                    if id in self.mouth_points:

                        x, y = int(lm.x*width), int(lm.y*height)
                        cv2.circle(frame, (x, y), 1, (255, 0, 255), cv2.FILLED)
                        mouth_points.append([x, y])
        return mouth_points

    def getYawnScore(self, mouthpoints):
        A = dist.euclidean(mouthpoints[8], mouthpoints[28])
        B = dist.euclidean(mouthpoints[2], mouthpoints[22])
        C = dist.euclidean(mouthpoints[29], mouthpoints[38])
        mar = (A+B)/2*C
        return mar/1000

    def warning(self, frame, mar_score):

        if mar_score > 1.3:
            cv2.putText(frame, "You are yawning", (50, 80),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            speak('You are yawning')
