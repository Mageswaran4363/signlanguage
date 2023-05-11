#-*- coding: utf-8 -*-
#import pyttsx3
import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator #google teachable machine trained by me
import tensorflow as tf

model = keras.models.load_model(r"C:\Users\Lakshmipriya\Desktop\sign\best_model.h5") #we used segmented dataset ,asl dataset


#word_dict = {6:"call",0:"Doctor",1:"Help",2:"Hot",3:"Lose",4:"Pain",5:"Theif"} computer to detect
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'O',14:'P',15:'Q',16:'R',17:'U',18:'V',19:'W',20:'Y',21:'Z'}




background = None
accumulated_weight = 0.5 #box values 

ROI_top = 100         #10
ROI_bottom = 300       #350
ROI_right = 150        #10
ROI_left = 350         #350

def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25): #backgroud object shape and turns background into black
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame) #unknow integer 

    
    _ ,thresholded = cv2.threshold(diff, threshold, 255,cv2.THRESH_BINARY)
    
     #Fetching contours in the frame (These contours can be of hand
#or any other object in foreground) …

    contours, hierarchy =cv2.findContours( thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any
   # contours...
    if len(contours) == 0:
        return None 
        
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the
 # thresholded image of hand...
        return (thresholded, hand_segment_max_cont)

def play():
    converter.say(words)
    converter.runAndWait()


cam = cv2.VideoCapture(0) #camera on
num_frames =0
#converter = pyttsx3.init()
while True:
    ret, frame = cam.read() #reading camera

    # flipping the frame to prevent inverted image of captured
   # frame...
    
    frame = cv2.flip(frame, 1) #image fliiping while showing 

    frame_copy = frame.copy() #camera copying the image

    # ROI from the frame
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #image colour  covert
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)


    if num_frames < 70:
        
        cal_accum_avg(gray_frame, accumulated_weight)
        
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2) #
    
    else: 
        # segmenting the hand region
        hand = segment_hand(gray_frame)
        
        # Checking if we are able to detect the hand...
        if hand is not None:
            
            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)
            
            cv2.imshow("Thesholded Hand Image", thresholded)
            
            thresholded = cv2.resize(thresholded, (64, 64))
            thresholded = cv2.cvtColor(thresholded,cv2.COLOR_GRAY2RGB)
            thresholded = np.reshape(thresholded,(1,thresholded.shape[0],thresholded.shape[1],3))# converting array to shape
            
            pred = model.predict(thresholded) # it detects which object
            cv2.putText(frame_copy, word_dict[np.argmax(pred)],(170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            words = ""     # it calls and detct what object naming
            words = word_dict[np.argmax(pred)]
            print(words)
            #if words:
            #    play()
            
            
            
            
    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,
    ROI_bottom), (255,128,0), 3) #box rectngle image

    # incrementing the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.putText(frame_copy, "hand sign recognition",
    (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    cv2.imshow("Sign Detection", frame_copy)
    print()

    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows() #ends 