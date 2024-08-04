import cv2
import numpy as np
import pygame

pygame.mixer.init()
alert_sound = pygame.mixer.Sound('C:/Users/sooda/Desktop/alarm.wav')#change file path here


cap = cv2.VideoCapture(0)

first_frame = None
motion_detected = False

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    diff = cv2.absdiff(first_frame, gray)

    thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)


    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    if motion_detected:
        alert_sound.play()
        motion_detected = False
    cv2.imshow("Motion Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
