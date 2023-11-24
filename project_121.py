import cv2
import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3, 640)
camera.set(4, 480)

# loading the mountain image
mountain = cv2.imread('IMG1.jpeg')

# resizing the mountain image as 640 X 480
mountain_1 = cv2.resize(mountain, (640, 480))

while True:
    # read a frame from the attached camera
    status, frame = camera.read()

    # if we got the frame successfully
    if status:
        # flip it
        frame = cv2.flip(frame, 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([100, 100, 100])
        upper_bound = np.array([255, 255, 255])

        # thresholding image
        mask_1 = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # inverting the mask
        mask_2 = cv2.bitwise_not(mask_1)

        # Apply the mask to the mountain image to replace the background
        mountain_masked = cv2.bitwise_and(mountain_1, mountain_1, mask=mask_2)

        # Apply the inverse mask to the original frame to keep the face
        frame_masked = cv2.bitwise_and(frame, frame, mask=mask_1)

        # Combine the masked mountain image and the masked frame
        final_output = cv2.add(frame_masked, mountain_masked)

        # show it
        cv2.imshow('frame', final_output)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
