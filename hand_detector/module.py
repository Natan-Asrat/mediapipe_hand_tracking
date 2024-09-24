import cv2
import mediapipe as mp
import time

class HandTracking():
    def __init__(self, static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5) -> None:
        self.static_image_mode=static_image_mode,
        self.max_num_hands=max_num_hands,
        self.model_complexity=model_complexity,
        self.min_detection_confidence=min_detection_confidence,
        self.min_tracking_confidence=min_tracking_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            max_num_hands=max_num_hands,
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, img, points=[0], draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        landmarks = self.results.multi_hand_landmarks
        if landmarks and draw:
            for landmark in landmarks:
                for id, point in enumerate(landmark.landmark):
                    h, w, c = img.shape
                    cx, cy = int(point.x * w), int(point.y * h)
                    if id in points and draw is True :
                        cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)
                
                self.mpDraw.draw_landmarks(img, landmark, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img):
        lmList = []
        landmarks = self.results.multi_hand_landmarks
        if landmarks:
            for hand in landmarks:
                handList=[]
                for id, point in enumerate(hand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(point.x * w), int(point.y * h)
                        handList.append([id, cx, cy])
                        
                lmList.append(handList)
        return lmList
                

def main():
    cap = cv2.VideoCapture(0)
    detector = HandTracking()
    while True:
        success, img = cap.read()
        drawn_img = detector.findHands(img)
        lmList = detector.findPosition(drawn_img)
        if len(lmList) != 0:
            print(lmList)
        cv2.imshow('Image', drawn_img)
        cv2.waitKey(1)



if __name__ == '__main__':
    main() 