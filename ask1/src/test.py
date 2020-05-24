import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import detector_test as d

Video_path = 'data/VIRAT_S_010000_03_000442_000528.mp4'
display_video = False
# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 2400, qualityLevel = 0.12, minDistance = 2, blockSize = 7, useHarrisDetector=1)
# Parameters for Harris corner detection
h_feature_params = dict(blockSize = 6, ksize = 3, k=0.18)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (10,10), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture('data/VIRAT_S_010000_03_000442_000528.mp4')

color = (0, 0, 255)

ret, first_frame = cap.read()

first_frame = cv2.resize(first_frame, (int(first_frame.shape[1]/2), int(first_frame.shape[0]/2)))
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
# prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
prev = d.detect_corners(prev_gray, feature_params, type="harris")
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)
frame_index = 1
while(cap.isOpened()):

    ret, frame = cap.read()
    frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculates sparse optical flow by Lucas-Kanade method
    #print(frame)
    if frame_index % 10 == 0:
        prev = d.detect_corners(prev_gray, feature_params, type="harris")
    next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    #print(next)
    # Selects good feature points for previous position
    good_old = prev[status == 1]

    # Selects good feature points for next position
    good_new = next[status == 1]

    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # a, b = coordinates of new point
        a, b = new.ravel()

        # a, b = coordinates of old point
        c, d = old.ravel()

        # Draws line between new and old position with green color and 2 thickness
        # Draws line between new and old position with red color and 2 thickness
        #if abs(a-c)>0 or abs(b-d)>0:
        if abs(a-c) + abs(b-d) >0.9:
            mask = cv2.line(mask, (a, b), (c, d), color, 2)

        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        #frame = cv2.circle(frame, (a, b), 3, color, -1)

    # Overlays the optical flow tracks on the original frame
    output = cv2.add(frame, mask)

    # Updates previous frame
    prev_gray = gray.copy()

    # Updates previous good feature points
    prev = good_new.reshape(-1, 1, 2)

    # Opens a new window and displays the output frame
    cv2.imshow("sparse optical flow", output)

    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# The following frees up resources and closes all windows

cap.release()
cv2.destroyAllWindows()
