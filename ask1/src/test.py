import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import detector_test as DT
import draw


Video_path = 'data/VIRAT_S_010000_03_000442_000528.mp4'
Video_out_path =''
display_video = False
# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 2400, qualityLevel = 0.12, minDistance = 2, blockSize = 7, useHarrisDetector=1)
# Parameters for Harris corner detection
h_feature_params = dict(blockSize = 6, ksize = 3, k=0.18)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (10,10), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv2.VideoCapture('data/VIRAT_S_010000_03_000442_000528.mp4')

color = (0, 0, 255)

save_vid=False
type="st"
fps = cap.get(cv2.CAP_PROP_FPS)
Nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.resize(prev_gray, (int(first_frame.shape[1]/2), int(first_frame.shape[0]/2)))
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive

prev_points = DT.detect_corners(prev_gray, feature_params, type)
# print(prev_points)
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)
mask = cv2.resize(mask, (int(mask.shape[1]/2), int(mask.shape[0]/2)))
#time_mask = np.zeros_like(mask)
#time_mask[prev_points.reshape((prev_points.shape[0],prev_points.shape[2]))] = 1

#Create video writer in order to store the video if save_vid == True
size = (prev_gray.shape[1],prev_gray.shape[0])
out_vid = cv2.VideoWriter(Video_out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

# recompute features
frame_index =  1
while (cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < Nframe):

  # Frame reading and processing of size and color space
  ret, frame = cap.read()

  frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  if frame_index % 30 == 0:
      prev_points = DT.detect_corners(prev_gray, feature_params, type)
      mask = np.zeros_like(mask)
      #time_mask[prev_points.reshape((prev_points.shape[0],prev_points.shape[2]))] += 1
  # Lucas Kanade algorithm
  next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points.astype(np.float32), None, **lk_params)

  # Selects good feature points for previous position
  good_old = prev_points[status == 1]
  # Selects good feature points for next position
  good_new = next[status == 1]

  frame, mask = draw.draw_optfl(good_new, good_old, frame.copy(),(mask), only_motion=True)
  #time_mask[time_mask > 0] += 1
  #mask = draw.update(mask, time_mask)
  # Overlays the optical flow tracks on the original frame
  output = cv2.add(frame, mask)
  # Updates previous frame
  prev_gray = gray.copy()
  # Updates previous good feature points
  prev_points = good_new.reshape(-1, 1, 2)

  frame_index += 1
  # Opens a new window and displays the output frame
  cv2.imshow("sparse optical flow", output)
  if save_vid:
      out_vid.write(output)

  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
