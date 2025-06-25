import cv2
import numpy as np

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_data = []
i = 0

while True:
    ret, frame = video.read() #Continuously reads frames from the webcam.
    
    #convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facedetect.detectMultiScale(gray, 1.3, 5) #Detects faces in the frame.

    #process each face detected
    for (x, y, w, h) in faces:

        crop_img =frame[y:y+h, x:x+w, :] #Crop the face from the frame.

        resized_img = cv2.resize(crop_img, (50, 50)) #Resize the cropped face to a fixed size.

        if len(face_data) <= 100 and i % 10 == 0:
            face_data.append(resized_img)
        i += 1

        cv2.putText(frame, str(len(face_data)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50,225), 1)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (50,50,225),1) #Draw rectangle around the detected face.
    
    cv2.imshow("Face Detection", frame) #Display the frame with detected faces.

    k = cv2.waitKey(1)#Wait for a key press.

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
    
