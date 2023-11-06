# import the modules
from genericpath import isfile
import os
import glob
from pathlib import Path
from posixpath import join
import cv2
import numpy as np

mypath='c:/Users/Kalyan/Desktop/TSK/Tarun/Tetrapak projects/Simple Straw Defect image classification/Straw Data'
onlyfiles = [ f for f in os.listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread( join(mypath,onlyfiles[n]) ) 
        src_gray = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        src_gray = cv2.blur(src_gray, (3,3))
        cv2.imshow('Gray Scale Pic', src_gray)

        # Apply thresholding to create a binary image
        _, thresh = cv2.threshold(src_gray, 70, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow('thresh', thresh)
        
        canny_output = cv2.Canny(src_gray, 100, 255)
    
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a list to store dark blob coordinates
        dark_blob_coordinates = []
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        # Define a threshold area to filter out small contours
        min_blob_area = 5
        Defect_Detected = False

        # Iterate through the contours and filter out small ones
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_blob_area:
            # Get the centroid of the contour
                #print(area)
                Defect_Detected = True
            
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
        
        drawing = np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)
        
        if Defect_Detected:
            cv2.putText(img=images[n], text='Not Good', org=(0, 70), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=(0, 0, 255),thickness=3)
            image_copy = images[n].copy()
            cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            #cv2.imshow('Dark Blobs', image_copy)
            # take the first contour
            cnt = contours[0]
            # compute the bounding rectangle of the contour
            x,y,w,h = cv2.boundingRect(cnt)
            # draw contour
            images[n] = cv2.drawContours(images[n],[cnt],0,(0,0,255),2)
            # draw the bounding rectangle
            images[n] = cv2.rectangle(images[n],(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow('contours of the defective straw', images[n])
            print("Defective straw found!")
        else:
            cv2.putText(img=images[n], text='Good', org=(0,70), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
            print("not Defective!")
            #cv2.imshow('no defect found', images[n])
            cv2.waitKey(0)
        
        for i in range(len(contours)):
                color = (0, 0, 255)
                drawing = cv2.drawContours(drawing, contours_poly, i, color)
                drawing = cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        if Defect_Detected:        
                cv2.imshow('highlighting the defect of straw', drawing)
                cv2.waitKey(0)
        # Display the original image with detected blobs
        cv2.destroyAllWindows()