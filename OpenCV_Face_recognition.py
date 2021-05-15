import cv2 # importing module

capture_video = cv2.VideoCapture(0) # 0=defult video source which is the computers webcam
cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_defult.xml') # prebuilt models to
# detect different objects

while True:
    ret,frame = capture_video.read() # reading the first frame from the video
    gray = cv2.cvtColor(frame,0)
    detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5) # the method used to detect the face in images
    # this method is from the cascade classifier class
    if (len(detections)>0): # this detector will give us all possible regions in the
        # respectvie image, if it has detected a face
        # the rectangle are the specific features the camera detects
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('frame',frame) # displaying the frame with the .imshow method
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break # stopping the program from running without causing KeyboardInterrupt error

    #cv2.waitKey(1): delaying each frame by 1 millisecond
capture_video.release()
cv2.destroyAllWindows()
