import cv2
import mediapipe as mp



#Initailizing the mediapipe pose estimation
cap=cv2.VideoCapture("bicep.mp4")

mpPose=mp.solutions.pose
pose=mpPose.Pose()

mpDraw=mp.solutions.drawing_utils




while True:
    ref,frame=cap.read()

    if ref:
        #mediapipe accepts the rgb images as the input
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=pose.process(frameRGB)

        lmlist=[]
        if results.pose_landmarks:
            mpDraw.draw_landmarks(frame,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h,w,c=frame.shape
                print(id,lm)

                cx,cy=int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])




            cv2.circle(frame,(lmlist[14][1],lmlist[14][2]),8,(255,0,255),cv2.FILLED)

                #print(id,lm)
        #cv2.imshow("video",frame)


        cv2.imshow("video",frame)
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break

    else:
        break


