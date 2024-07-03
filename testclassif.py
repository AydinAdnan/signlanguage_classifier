#test
import cv2
import mediapipe as mp
import pickle
import numpy as np
import logging
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

model_dict=pickle.load(open('./model.p','rb'))
model=model_dict['model']

cap=cv2.VideoCapture(0)

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)
labels_dict={0:'One',1:'Two',2:'Three',3:'Four',4:'Five'}
while True:
    data_aux=[]
    x_=[]
    y_=[]
    ret,frame=cap.read()
    H,W,_=frame.shape
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results=hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                    frame, #image to draw
                    hand_landmarks, #model output
                    mp_hands.HAND_CONNECTIONS, #connections between landmarks
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x= hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
        x1=int(min(x_)*W)
        y1=int(min(y_)*H)
        x2=int(max(x_)*W)
        y2=int(max(y_)*H)


        prediction=model.predict([np.asarray(data_aux)])
        predicted_character=labels_dict[int(prediction[0])]
        print(predicted_character)
    
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),4) #black is (0,0,0) and thickness 4
        cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)


    cv2.imshow('frame',frame)
    cv2.waitKey(500) #wait 25ms b/w each frame




cap.release()
cv2.destroyAllWindows()