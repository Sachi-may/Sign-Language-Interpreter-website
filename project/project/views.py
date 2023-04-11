from django.http import HttpResponse
from django.shortcuts import render
import  cv2
import numpy as np
import os
import time
#from keytotext import pipeline
import mediapipe as mp
from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping
 
def sign_language(request):
    return render(request, 'index.html')

def aboutus(request):
    return render(request, 'aboutus.html')

def sachis(request):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    sequence  = []
    sentence = []
    predictions = []
    threshold = 0.5
    cap = cv2.VideoCapture(0)
    def mediapipe_detection(image,model):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        return image, results
    def extract_keypoints(results):
        pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        left_hand = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
        right_hand = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
        return np.concatenate([pose,left_hand,right_hand])
    actions = np.array(['yes','no','hello','Thank you','learn','understand','you','I','sign','what','name','like'])
    no_sequences = 30
    sequence_length = 30
    start_folder = 30


    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape =(30,258)))
    # model.add(GRU(128, return_sequences=True, activation='relu'))
    model.add(GRU(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32,  activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()


    model.load_weights(r'C:\Users\sachi\Downloads\major_data\model11.h5')

    colors = [(245,117,16), (117,245,16), (16,117,245),(245,16,117),(117,16,245),(16,245,117),(100,245,50),(100,50,245),(50,100,245),(50,245,100),(245,100,50),(245,50,100)]
    def prob_viz(res, actions, input_frame):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,10+num*40), (int(prob*100), 40+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 35+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)        
        return output_frame
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():        
            ret, img = cap.read()        
            image, results = mediapipe_detection(img, holistic)        
            #
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]  
                predictions.append(np.argmax(res)) 
                if np.unique(predictions[-10:])[0]==np.argmax(res):        
                    if res[np.argmax(res)] > threshold:                   
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                    

                if len(sentence) > 5: 
                    sentence = sentence[-5:]
                image = prob_viz(res, actions, image)
            
            cv2.imshow('Video', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    return render(request, 'index.html')

