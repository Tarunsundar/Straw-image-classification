# import the modules
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours
import matplotlib.pyplot as plt
import numpy as np
import argparse
from genericpath import isfile
import os
from posixpath import join
from timeit import default_timer as timer
import cv2
import numpy as np
from datetime import datetime

mypath='c:/Users/Kalyan/Desktop/TSK/Tarun/Tetrapak projects/Simple Straw Defect image classification/Straw Data'
onlyfiles = [ f for f in os.listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
startTime = timer()
directory = 'C:/Users/Kalyan/Desktop/TSK/Tarun/Tetrapak projects/Simple Straw Defect image classification/Defective_Straws'

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread( join(mypath,onlyfiles[n]) ) 
        gray = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        gray = cv2.blur(gray, (3,3))
        
        src_gray = cv2.blur(gray, (3,3))
        
        # Apply thresholding to create a binary image
        _, thresh = cv2.threshold(src_gray, 80, 255, cv2.THRESH_BINARY_INV) #minimum threshold 120, ideal

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Define a threshold area to filter out small contours
        min_blob_area = 5
        defect_detected = False
        straw_detected = False
        straw_count = 0
        blob_count = 0 
        total_defect_area= 0
        # Change the current directory  
        # to specified directory  
        os.chdir(directory) 
        c2 = None
        c2_1 = None
        c2_2 = None

        # Iterate through the contours and filter out small ones
        for contour in contours:
            area = cv2.contourArea(contour)
            blob_count = len(contours)
            x,y,w,h = cv2.boundingRect(contour)
            minArea= 25000
            maxArea= 90000
            c1 = contour[0]
            c1_1 = c1[0][0]
            c1_2 = c1[0][1]
            if(c2 is None):
                 c2 = contour[0]  
                 c2_1 = c2[0][0]
                 c2_2 = c2[0][1]
            dx = abs(np.subtract(c1_1, c2_1))
            dy = abs(np.subtract(c1_2, c2_2))
            if int(area) in range(minArea, maxArea ,1):  #range could be of a straw
                straw_count = straw_count + 1
                print("straw number: ", straw_count)
                straw_detected = True
                blob_count+=1
                if(dx < 40 and dy>3 or h>100):
                    defect_detected = True
                    total_defect_area += area
                    cv2.drawContours(images[n], contour,-1,[0,0,255],2)
                    cv2.putText(images[n],text='defective straw area = ' +str(area)+' width & height = '+str(w)+","+str(h), org=(x,y-5), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,0,255),thickness = 1)
                else:
                     cv2.drawContours(images[n], contour,-1,[0,255,0],2)
                     cv2.putText(images[n],text='straw area = ' +str(area)+' width & height = '+str(w)+","+str(h), org=(x,y-5), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,255,0),thickness = 1)
            else:
                  blob_count+=1
                  defect_detected = True
                  total_defect_area += area
                  x,y,w,h = cv2.boundingRect(contour)
                  cv2.drawContours(images[n], contour,-1,[0,160,255],2)
            
            if(c2 is not None): 
                c2 = c1[0]
                c2_1 = c1[0][0]
                c2_2 = c1[0][1]
            
        darkBlobs = np.zeros((thresh.shape[0], src_gray.shape[1], 3), dtype=np.uint8)
        cv2.putText(images[n], text = 'straw count = '+str(straw_count), org=(0,50), fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0, 0, 255),thickness=1)
        if defect_detected:
            cv2.putText(images[n], text = 'defect count = '+str(blob_count), org=(0,70), fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0, 0, 255),thickness=1)
            cv2.putText(img=images[n], text='defect detected, total defect area(in pexels) = ' + str(total_defect_area), org=(0,30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 255),thickness=1)
            cv2.resize(images[n], (960, 540))
            cv2.imshow('Detected defective contours',images[n])
            now = datetime.now()
            current_time = now.strftime("%H%M%S")
            filename = "defective straw@" + str(current_time) + ".bmp"
            cv2.imwrite(filename, images[n])
            cv2.waitKey(0)
        else:
            cv2.putText(images[n], text= "no defect detected", org=(0,50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0),thickness=1)
            cv2.resize(images[n], (960, 540))
            cv2.imshow('straws without defects', images[n]) #remove this if not neccessary 
            cv2.waitKey(0)

        # Display the original image with detected blobs
        cv2.destroyAllWindows()
