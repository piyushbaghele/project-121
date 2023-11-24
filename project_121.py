# import cv2 to capture videofeed
import cv2

import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('IMG1.jpeg')

# resizing the mountain image as 640 X 480
mountain_1 = cv2.resize(mountain,(640,480))

while True:

    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame , 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_red = np.array([100,100,100])
        upper_red = np.array([255, 255,255])

        # thresholding image
        mask_1 = cv2.inRange(frame_rgb ,lower_red, upper_red)

        # inverting the mask
        mask_1= cv2.morphologyEx(mask_1, cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
       
        # bitwise and operation to extract foreground / person
        mask_2 = cv2.bitwise_not(mask_1)
        # final image
        mask_2_resized = cv2.resize(mask_2, (640, 480))
        


        print("mountain_1 shape:", mountain_1.shape)
       
        mask_2_resized_color = cv2.cvtColor(mask_2_resized, cv2.COLOR_GRAY2BGR)
        print("mask_2_resized shape:", mask_2_resized_color.shape)
        final_output = cv2.add(mountain_1, mask_2_resized_color)
        # show it
        cv2.imshow('frame' , final_output)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
