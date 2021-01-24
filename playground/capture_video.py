import cv2
import datetime
cap = cv2.VideoCapture(0) # object captured video from web cam
fourcc = cv2.VideoWriter_fourcc(*'XVID') # fourcc code given
# http://www.fourcc.org/codecs.php 
# arguments: filename, fourcc code, fps, size of the video frame
out = cv2.VideoWriter('output.avi', fourcc, 24, (640,480))
# setting parametra, width and height
cap.set(3, 1920)
cap.set(4, 1080)

print(cap.get(3))
print(cap.get(4))
while cap.isOpened(): # while caoture object is exist
    # ret - bool var
    ret, frame = cap.read()
    if ret:
        font = cv2.FONT_HERSHEY_SIMPLEX
        datet = str(datetime.datetime.now())
        frame = cv2.putText(frame, datet, (10, 500), fontFace=font, fontScale=1, 
                            color=(255,255,255), thickness=5, lineType=cv2.LINE_AA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # change colors to gray scale
        # get height and width of the frame
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # write the frame to the file
        # out.write()
        # another properties availeble here:
        # https://docs.opencv.org/4.0.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'): # need to specify how to quit the win frame
            break
    else:
        break
# realese and break window
cap.release()
out.release()
cv2.destroyAllWindows()
