import cv2
import poseestimationmodule as pm
import time

cap=cv2.VideoCapture("bicep.mp4")

detector=pm.poseDetector()

ptime=0

while True:
    ret,frame=cap.read()
    if ret:
        frame=detector.findPose(frame)
        lmlist=detector.findPosition(frame)
        #print(lmlist)
        if lmlist!=0:
            cv2.circle(frame,(lmlist[26][1],lmlist[26][2]),10,(255,100,100),cv2.FILLED)

        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        pos=[130,130]
        ox,oy=pos[0],pos[1]
        offset=10
        text=str(int(fps))

        (w,h),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_PLAIN,3,3)
        x1,y1,x2,y2=ox-offset,oy+offset,ox+w+offset,oy-h-offset
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(frame,text,(ox,oy),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)


        cv2.imshow("video",frame)
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break

    else:
        break

