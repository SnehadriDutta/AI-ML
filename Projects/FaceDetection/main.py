import os
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
import tensorflow as tf


face_cascade = cv2.CascadeClassifier('venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')
model = tf.keras.models.load_model("C:\\Users\\hp\\Desktop\\Misc\\Programming\\Machine & Deep Learning\\FaceRecognition\\face-recognization.h5")

label_id = {0 : "Cristiano Ronaldo",
           1 : "Elon Musk",
           2 : "Lionel Messi",
           3 : "Sachin-Tendulkar",
           4 : "Snehadri Dutta"}

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter("MyVideo.avi", cv2.VideoWriter_fourcc(*"XVID"), 5, (640,480))

while (True):

    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor= 1.5, minNeighbors= 5)
    for (x,y,w,h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]     # region of interest
        roi_color = frame[y:y + h, x:x + w]

        face = cv2.resize(frame, (244, 244))
        im = Image.fromarray(face, 'RGB')
        # Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
        # Our keras model used a 4D tensor, (images x height x width x channel)
        # So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)

        name = "New Face"
        if (pred[0][0] > 0.5):
            name = label_id[0]
        elif (pred[0][1] > 0.5):
            name = label_id[1]
        elif (pred[0][2] > 0.5):
            name = label_id[2]
        elif (pred[0][3] > 0.5):
            name = label_id[3]
        elif (pred[0][4] > 0.5):
            name = label_id[4]

        font = cv2.FONT_HERSHEY_SIMPLEX
        color_text = (255,0,0)
        #name = "Snehadri-Dutta"
        stroke_text = 2
        cv2.putText(frame, name, (x,y-10), font, 0.75, color_text, stroke_text, cv2.LINE_AA)


        color = (0,0,255) # BGR
        stroke = 2
        start_x = x
        start_y = y
        end_x = x + w
        end_y = y + h

        cv2.rectangle(frame, (start_x,start_y), (end_x, end_y), color, stroke)

    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
