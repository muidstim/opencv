from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

try:
    # use the trainer.yml file that was created with face_trainer.py
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    # id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc.
    names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W'] 

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 24
    camera.rotation = 180
    rawCapture = PiRGBArray(camera, size=(640, 480))
     
    # allow the camera to warmup
    time.sleep(0.1)
    
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
            
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
        
            cv2.putText(img, str(id), (x + 5,y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)  
    
        cv2.imshow('camera',img) 
        
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
          break            
          
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

finally:
    cv2.destroyAllWindows()
