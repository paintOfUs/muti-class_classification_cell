import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

url = "D:\\important_for_study_master\\DoAn\\New folder\\data\\230912\\2C0\\36.jpg"
img = cv.imread(url)
# Convert BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([40,90,90])
upper_blue = np.array([166,255,255])
# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_blue, upper_blue)
mask = cv.GaussianBlur(mask,(5,5),0)

# Create a mask for white and red colors
lower_white = np.array([0, 0, 255])
upper_white = np.array([255, 40, 255])
mask_white = cv.inRange(hsv, lower_white, upper_white)

# Combine the blue and white masks to keep both blue and white/red areas
final_mask = cv.bitwise_or(mask, mask_white)

# Bitwise-AND mask and original image
res = cv.bitwise_and(img,img, mask= final_mask)
cv.imshow('frame',img)
cv.imshow('mask',mask)
cv.imshow('res',res)

ret,thresh = cv.threshold(final_mask,127,255,0)

contours,hierachy=cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# Draw bounding boxes around the detected contours
for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

# Display the image with bounding boxes

cv.imshow('bouding box', img)

cv.waitKey(0)
cv.destroyAllWindows()

