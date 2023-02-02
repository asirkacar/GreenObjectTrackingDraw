import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cizim = False

a, b= -1,-1
bosEkran = np.zeros((480,640,3), np.uint8)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower = np.array([0, 109, 219])
    upper = np.array([216, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    _, esik = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
    contur,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for k in contur:
        (x,y,w,h) = cv2.boundingRect(k)
        if cv2.contourArea(k) > 700:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            centX, centY = ((x+(x+w)) / 2, (y+(y+h)) / 2)
            a, b = int(centX), int(centY)
            cv2.putText(frame, "center", (int(centX) - 20, int(centY) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cizim = True
            if cizim == True:
                #cv2.line(frame, (a, b), (int(centX), int(centY)), (255,0,0), 5)
                cv2.circle(frame, (int(centX), int(centY)), 10, (255, 0, 0), -2)

                cv2.circle(bosEkran, (int(centX), int(centY)), 10, (255, 0, 0), -2)
                f,g,h = bosEkran.shape
                roi = frame[0:f, 0:g]
                frameGray = cv2.cvtColor(bosEkran, cv2.COLOR_BGR2GRAY)
                ret, mask2 = cv2.threshold(frameGray, 20, 255, cv2.THRESH_BINARY)
                mask2_inv = cv2.bitwise_not(mask2)
                bosEkran_bg = cv2.bitwise_and(bosEkran, bosEkran, mask=mask2)
                bosEkran_fg = cv2.bitwise_and(roi, roi, mask=mask2_inv)
                toplam = cv2.add(bosEkran_fg,bosEkran_bg)
                frame[0:f, 0:g] = toplam

                print(a, b)
            else:
                cizim = False
    cv2.imshow("Image", frame)
    cv2.imshow("bosE", bosEkran)
    #cv2.imshow("bos", bosEkran)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()