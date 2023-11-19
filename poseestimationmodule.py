import cv2
import mediapipe as mp

class poseDetector():
    def __init__(self,mode=False,mcomplexity=1,slandmarks=True,esegmentation=False,
                 ssegmentation=True,detconfidence=0.5,trackconfidence=0.5):
        self.mode=mode
        self.mcomplexity=mcomplexity
        self.slandmarks=slandmarks
        self.esegmentation=esegmentation
        self.ssegmentation=ssegmentation
        self.detconfidence=detconfidence
        self.trackconfidence=trackconfidence
        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.mcomplexity,self.slandmarks,
                              self.esegmentation,self.ssegmentation,self.detconfidence,self.trackconfidence)


    def findPose(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)


        return img
def main():

    cap=cv2.VideoCapture("bicep.mp4")
    detector=poseDetector()
    while True:
        ret,frame=cap.read()
        if ret:
            frame=detector.findPose(frame)
            cv2.imshow("video",frame)
            if cv2.waitKey(1) & 0xFF==ord('1'):
                break

        else:
            break


if __name__=='__main__':
    main()


