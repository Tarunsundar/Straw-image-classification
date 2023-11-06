# import the modules
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours
import numpy as np
from datetime import timedelta
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

""" #this should be added while reading images from a camera
cap = cv2.VideoCapture(0)
# Capture the video frame
# by frame
while(True): #this should be replaced with the for loop

ret, frame = cap.read()
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = cap.get(cv2.CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

""" 
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
        
for n in range(0, len(onlyfiles)): #remove this when using a camera
        images[n] = cv2.imread( join(mypath,onlyfiles[n]) ) #replace images with frame
        src_gray = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        src_gray = cv2.blur(src_gray, (3,3))
        
        # Apply thresholding to create a binary image
        _, thresh = cv2.threshold(src_gray, 80, 255, cv2.THRESH_BINARY_INV) #minimum threshold 120, ideal

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Define a threshold area to filter out small contours
        min_blob_area = 1
        defect_detected = False
        straw_detected = False
        straw_count = 0
        blob_count = 0 
        total_defect_area= 0
        
        # Change the current directory  
        # to specified directory  
        os.chdir(directory) 

        # Iterate through the contours and filter out small ones
        for contour in contours:
            area = cv2.contourArea(contour)
            blob_count = len(contours)
            x,y,w,h = cv2.boundingRect(contour)
            minArea= 28000
            maxArea= 46000
            minH= 50
            maxH= 100
            cv2.drawContours(images[n], contours, -1, (0, 255, 0), 1) 
            # repleace area with dimA
            if int(area) in range(minArea, maxArea,1) and h in range(minH,maxH,1):  #range should be between 36000 to 43000
            # Get the centroid of the contour
                straw_count = straw_count + 1
                straw_detected = True
                cv2.rectangle(images[n],(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(images[n],text='straw area = ' +str(area)+' width & height = '+str(w)+","+str(h), org=(x,y-10), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,255,0),thickness = 1)
            elif int(area) in range(30000,34999,1) or area > 46000 or h > 85:
                 total_defect_area += area
                 cv2.rectangle(images[n],(x,y),(x+w,y+h),(0,120,255),2)   
                 cv2.putText(images[n],text='defective straw area = ' +str(area)+' width & height = '+str(w)+","+str(h), org=(x,y-10), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,120,255),thickness = 1)
            else:
                  defect_detected = True
                  total_defect_area += area
                  #print("area of the defect detected = ", area)
                  x,y,w,h = cv2.boundingRect(contour)
                  cv2.rectangle(images[n],(x,y),(x+w,y+h),(0,0,255),2)
            blob_count= blob_count - straw_count
            
        darkBlobs = np.zeros((thresh.shape[0], src_gray.shape[1], 3), dtype=np.uint8)
        
        if defect_detected:
            cv2.putText(images[n], text = 'defect count = '+str(blob_count), org=(0,70), fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0, 0, 255),thickness=1)
            cv2.putText(img=images[n], text='defect detected, total defect area(in pexels) = ' + str(total_defect_area), org=(0,30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 255),thickness=1)
            cv2.imshow('Detected defective contours',images[n])
            now = datetime.now()
            current_time = now.strftime("%H%M%S")
            filename = "defective straw@" + str(current_time) + ".bmp"
            cv2.imwrite(filename, images[n])
            cv2.waitKey(0)
        else:
             cv2.putText(images[n], text= "no defect detected", org=(0,50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0),thickness=1)
             cv2.imshow('straws without defects', images[n]) #remove this if not neccessary 
             cv2.waitKey(0)

        # Display the original image with detected blobs
        cv2.destroyAllWindows()

endTime= timer()
totalTime = float(endTime) - float(startTime)
print("total time taken to process image", totalTime/5)

"""
            pixelsPerMetric = None

            # compute the rotated bounding box of the contour
            orig = images[n].copy()
            box = cv2.minAreaRect(contour)
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
"""