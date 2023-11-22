import cv2
import time
import numpy as np
import poseestimationmodule as pm

cap=cv2.VideoCapture("bicep.mp4")
detector=pm.poseDetector()

while True:
    ret,frame=cap.read()
    if ret:
        frame=detector.findPose(frame)
        #print(lmlist)

        lmlist=detector.findPosition(frame,draw=False)
        print(lmlist)

        detector.findAngle(frame,11,13,15,draw=True)



        cv2.imshow("Video",frame)
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break

    else:
        break


