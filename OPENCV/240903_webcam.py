import cv2 as cv

#gst_str = ("filesrc location=bottle-detection.mp4 ! decodebin ! video/x-raw ! queue ! videoconvert ! appsink")
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print('fail')
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2LUV)

    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# Extra: Play video in terminal with the following command:
# gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! videoscale ! video/x-raw, width-640, height=360 ! autovideosink