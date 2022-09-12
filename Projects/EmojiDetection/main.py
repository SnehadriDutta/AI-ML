import numpy as np
import cv2
import re
import tensorflow as tf

dict_emojis = { 0: "emoji/angry.png",
                1: "emoji/disgust.png",
                2: "emoji/fear.png",
                3: "emoji/happy.png",
                4: "emoji/neutral.png",
                5: "emoji/sad.png",
                6: "emoji/surprise.png"}

emoji_model = tf.keras.models.load_model("emoji-detection-model.h5")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor= 1.5, minNeighbors=5)
    gray_frame = cv2.resize(gray_frame, (48,48))
    arr_img = np.asarray(gray_frame)
    arr_img = arr_img.reshape((arr_img.shape[0], arr_img.shape[1], 1))
    img_array = np.expand_dims(arr_img, axis=0)
    prediction = emoji_model.predict(img_array)

    index = np.unravel_index(prediction.argmax(), prediction.shape)[1]

    # Overlaying the emoji
    emoji = cv2.imread(dict_emojis[index])
    emoji = cv2.resize(emoji, (200, 200), interpolation=cv2.INTER_AREA)
    emoji = cv2.cvtColor(emoji, cv2.COLOR_BGR2BGRA)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_h, frame_w, frame_c = frame.shape

    overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')
    emoji_h, emoji_w, emoji_c = emoji.shape

    for i in range(0, emoji_h):
        for j in range(0, emoji_w):
            if emoji[i, j][3] != 0:
                offset = 10
                h_offset = frame_h - emoji_h - offset
                w_offset = frame_w - emoji_w - offset
                overlay[h_offset + i, w_offset + j] = emoji[i, j]

    cv2.addWeighted(overlay, 1, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    emotion = re.search('emoji/(.*).png', dict_emojis[index]).group(1)
    cv2.putText(frame, emotion, (frame_w-225, frame_h-225), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
