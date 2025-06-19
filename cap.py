import cv2

vdeo_path = 0
cap = cv2.VideoCapture(vdeo_path)
while cap.isOpened() :
    # Read a frame from the video
    success, frame = cap.read()
    cv2.imshow("kkk",frame)
    cv2.waitKey(1)
cap.release()