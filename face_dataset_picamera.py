from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    face_id = input('\n Enter user id end press <return> ==>  ')
    print("\n [INFO] Initializing face capture. Look the camera and press 'c' to take each photo.")


    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 24
    camera.rotation = 180
    rawCapture = PiRGBArray(camera, size=(640, 480))
     
    # allow the camera to warmup
    time.sleep(0.1)
    
    # Start a counter to keep track of how many photos we have taken
    count = 0
    
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image
        img = frame.array

        # convert to gray scale for the face detect algorithm
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # use the haar cascade to detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # loop through all the faces found and draw rectangles around them
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # save just the face roi on the gray image
            roi_gray = gray[y:y + h, x:x + w]
            #roi_color = img[y:y + h, x:x + w]
            
            # check for the user to pressed the 'c' key
            key = cv2.waitKey(100) & 0xff
            if key == ord("c"):
                # save the gray face to a file in the dataset folder called User1.jpg, User2.jpg etc.
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", roi_gray)
                
                count += 1
        
        # show the frame
        cv2.imshow("Frame", img)
        
        key = cv2.waitKey(100) & 0xFF
 
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        elif count >= 30: # break from the loop after 30 faces have been saved
            break
	    
finally:
    cv2.destroyAllWindows()
