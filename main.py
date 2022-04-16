import cv2
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
# Define the codec and create VideoWriter object
#fourcc = cv.VideoWriter_fourcc(*'XVID')
#out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 90)
    # write the flipped frame
    #out.write(frame)

    #low_red = (7,40,60)
    #high_red = (18,255,200)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 85, 110], dtype="uint8")
    upper_red = np.array([15, 255, 255], dtype="uint8")

    # красный в диапазоне фиолетового оттенка
    lower_violet = np.array([165, 85, 110], dtype="uint8")
    upper_violet = np.array([180, 255, 255], dtype="uint8")

    red_mask_orange = cv2.inRange(frame, lower_red, upper_red)  # применяем маску по цвету
    red_mask_violet = cv2.inRange(frame, lower_violet, upper_violet)  # для красного таких 2

    red_mask_full = red_mask_orange + red_mask_violet

    #frame = cv2.inRange(frame, red_mask_orange, red_mask_violet)

    cv.imshow('frame', red_mask_full)
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
#out.release()
cv.destroyAllWindows()
