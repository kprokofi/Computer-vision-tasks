import math
import cv2
import numpy as np

def nothing(x):
    pass
def erase(chick):
    if chick:
        img2[:] = img.copy() 
# Load an image
img = cv2.imread('image_example/cats.jpg')
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# Create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)
cv2.createTrackbar('width','image',0,10,nothing)

# Create trackbars for drawing shapes
cv2.createTrackbar('Select', 'image',0,3,nothing)
cv2.createTrackbar('erase', 'image',0,1,erase)

drawing = False # true if mouse is pressed
ix,iy = -1,-1
# mouse callback function
def draw(event,x,y,flags,param):
    global ix,iy,drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if s:
                cv2.rectangle(img2,(ix,iy),(x,y),(b,g,r),-1)
            else:
                cv2.circle(img2,(x,y),rad,(b,g,r),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if s:
            cv2.rectangle(img2,(ix,iy),(x,y),(b,g,r),-1)
        else:
            cv2.circle(img2,(x,y),rad,(b,g,r),-1)

cv2.setMouseCallback('image',draw)
# img = np.zeros((1280,720,3), np.uint8)
img2 = img.copy()
while(1):
    cv2.imshow('image',img2)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
    	break
    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    rad = cv2.getTrackbarPos('width','image')
    s = cv2.getTrackbarPos('Select','image')
    e = cv2.getTrackbarPos('erase','image')

cv2.destroyAllWindows()