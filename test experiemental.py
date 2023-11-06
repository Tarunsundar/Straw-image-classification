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
startTime = timer()
# Check whether the specified path exists or not
isExist = os.path.exists(directory)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(directory)
   print("The new directory is created!")

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
            straw_detected = True
            blob_count = len(contours)
            x,y,w,h = cv2.boundingRect(contour)
            minArea= 25000
            maxArea= 90000
            dyMax = 10             
            if int(area) in range(minArea, maxArea ,1):  #range could be of a straw
                straw_count = straw_count + 1
                for cts in contour:
                    c1 = cts
                    c1_1 = c1[0][0]
                    c1_2 = c1[0][1]
                    if(c2 is None):
                        c2 = cts  
                        c2_1 = c2[0][0]
                        c2_2 = c2[0][1]
                    dx = abs(np.subtract(c1_1, c2_1))
                    dy = abs(np.subtract(c1_2, c2_2))   
                    if( (dx == 1 and dy >= dyMax) or h>96):
                        print("difference between contour in straw no: ", straw_count,', ',dx, ",", dy)
                        print(c1[0],c2)  
                        defect_detected = True
                        blob_count += 1                         
                        total_defect_area += area
                    if(c2 is not None): 
                        c2 = c1[0]
                        c2_1 = c1[0][0]
                        c2_2 = c1[0][1]
                if(defect_detected == False):
                     cv2.putText(images[n],text='straw area = ' +str(area)+' dx = ' +str(dx)+' dy = '+str(dy)+" height= "+str(h), org=(x,y-10), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,255,0), thickness=1)
                     cv2.drawContours(images[n], contour,-1,[0,255,0],2)                     
                else:
                     cv2.drawContours(images[n], contour,-1,[0,0,255],2)
                     cv2.putText(images[n],text='defective straw area = ' +str(area)+' dx = ' +str(dx)+' dy = '+str(dy)+" height= "+str(h), org=(x,y-10), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,0,255), thickness=1)
            else:
                blob_count+=1
                defect_detected = True
                total_defect_area += area
                x,y,w,h = cv2.boundingRect(contour)
                cv2.drawContours(images[n], contour,-1,[0,160,255],2) 
            
        darkBlobs = np.zeros((thresh.shape[0], src_gray.shape[1], 3), dtype=np.uint8)
        cv2.putText(images[n], text = 'straw count = '+str(straw_count), org=(0,50), fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0, 160, 255),thickness=1)

        if defect_detected:
            cv2.putText(images[n], text = 'defect count = '+str(blob_count), org=(0,70), fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0, 0, 255),thickness=1)
            cv2.putText(img=images[n], text='defect detected, total defect area(in pexels) = ' + str(total_defect_area), org=(0,30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 255),thickness=1)
            cv2.resize(images[n], (960, 540))
            #cv2.imshow('Detected defective contours',images[n])
            now = datetime.now()
            current_time = now.strftime("%H%M%S")
            filename = "defective straw@" + str(dyMax) + ".bmp"
            cv2.imwrite(filename, images[n])
            cv2.waitKey(0)
        else:
            cv2.putText(images[n], text= "no defect detected", org=(0,30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0),thickness=1)
            cv2.resize(images[n], (960, 540))
            #cv2.imshow('straws without defects', images[n]) #remove this if not neccessary 
            cv2.waitKey(0)

        # Display the original image with detected blobs
        cv2.destroyAllWindows()

endTime= timer()
totalTime = float(endTime) - float(startTime)
print("total time taken to process image", totalTime)
"""
# compute gradients along the x and y axis, respectively
gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
# compute the gradient magnitude and orientation
magnitude = np.sqrt((gX ** 2) + (gY ** 2))

# initialize a figure to display the input grayscale image along with
# the gradient magnitude and orientation representations, respectively
(fig, axs) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# find contours in the edge map
cnts = cv2.findContours(magnitude, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# plot each of the images
axs[0].imshow(gray, cmap="gray")
axs[1].imshow(magnitude, cmap="jet")

# set the titles of each axes
axs[0].set_title("Grayscale")
axs[1].set_title("Gradient Magnitude")

# loop over each of the axes and turn off the x and y ticks
for i in range(0, 2):
    axs[i].get_xaxis().set_ticks([])
    axs[i].get_yaxis().set_ticks([])



# show the plots
plt.tight_layout()
plt.show()

# Iterate through the contours and filter out small ones
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 200:
    continue
    area = cv2.contourArea(c)
    x,y,w,h = cv2.boundingRect(c)
    # compute the rotated bounding box of the contour
    orig = images[n].copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    # draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        (255, 0, 255), 2)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / w

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
    actualsize = int(dimA*dimB)
    

    if int(dimA) in range(50, 90,1) and h in range(50,85,1):  #if normal straw
    # Get the centroid of the contour
        straw_count = straw_count + 1
        straw_detected = True
        cv2.rectangle(images[n],(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(images[n],text="dimA "+ str(dimA)+" dimB "+str(dimB)+' width & height = '+str(w)+","+str(h), org=(x,y-10), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,255,0),thickness = 1)
    elif int(dimA) in range(20,54,1) or area > 46000 or h > 85: #if defective straw
        total_defect_area += area
        cv2.rectangle(images[n],(x,y),(x+w,y+h),(0,120,255),2)   
        cv2.putText(images[n],text="dimA "+ str(dimA)+" dimB "+str(dimB)+" (inches) " + ' width & height = '+str(w)+","+str(h), org=(x,y-10), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,120,255),thickness = 1)
        cv2.putText(orig, str(actualsize)+" (inches) "+str(w)+","+str(h) +" contour area = "+str(cv2.contourArea(c)),(0, 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 1)
    else:
        defect_detected = True
        total_defect_area += area
        #print("area of the defect detected = ", area)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(images[n],(x,y),(x+w,y+h),(0,0,255),2)
    
# show the output image
cv2.imshow("straw Image", orig)
cv2.waitKey(0)

# Destroy the original image with detected blobs
cv2.destroyAllWindows()

endTime= timer()
totalTime = float(endTime) - float(startTime)
print("total time taken to process image", totalTime/5)
"""