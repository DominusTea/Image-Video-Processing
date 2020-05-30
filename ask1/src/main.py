import numpy as np
import cv2
import detector_test as DT
import draw
import matplotlib.pyplot as plt
import surveillance as surv

if __name__=="__main__":

    Do_Part_4 = True
    Do_Part_5 = False
    Do_Part_6 = False
    Do_Part_7 = False
    AM =  '03116045'

    Video_path = 'data/VIRAT_S_010000_03_000442_000528.mp4'

    if Do_Part_4:
        #part 4. Appy Lucas-Kanade on video using st or harris corner detection
        display_video = False
        Video_out_path = 'data/out4_video.avi'
        save_vid = False

        # Parameters for Shi-Tomasi corner detection
        st_feature_params = dict(maxCorners = 2000, qualityLevel = 0.1, minDistance = 2, blockSize = 7)
        # Parameters for Harris corner detection
        #h_feature_params = dict(blockSize = 6, ksize = 3, k=0.18)
        h_feature_params = dict(maxCorners = 2400, qualityLevel = 0.08, minDistance = 2, blockSize = 7, useHarrisDetector=1)

        # Parameters for Lucas Kanade optical flow field algorithm
        lk_params = dict(winSize = (10,10), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        #cap = cv2.VideoCapture(Video_path)
        #surv.lk4(cap, "st", st_feature_params, lk_params, 'data/out4_st.avi',save_vid = True)
        cap = cv2.VideoCapture(Video_path)
        surv.lk4(cap, "st", st_feature_params, lk_params, 'data/out4_st_opt.avi', save_vid = True)

    #testing Parameters


#part 5. Better following of the points and occasional refreshing of the points we follow

    if Do_Part_5:

          feature_params = dict(maxCorners = 2400, qualityLevel = 0.08, minDistance = 2, blockSize = 7, useHarrisDetector=1)
          # Parameters for Harris corner detection
          # Parameters for Shi-Tomasi corner detection
          #st_feature_params = dict(maxCorners = 2000, qualityLevel = 0.1, minDistance = 5, blockSize = 7)
          st_feature_params = dict(maxCorners = 2000, qualityLevel = 0.1, minDistance = 2, blockSize = 7)

          # Parameters for Lucas Kanade optical flow field algorithm
          lk_params = dict(winSize = (10,10), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
          cap = cv2.VideoCapture(Video_path)
          surv.lk5(cap, "harris", feature_params, lk_params, 'data/out5_harris_opt.avi', save_vid = True)


      if Do_Part_6:

          # Parameters for Harris corner detection
          feature_params = dict(maxCorners = 2400, qualityLevel = 0.08, minDistance = 2, blockSize = 7, useHarrisDetector=1)
          # Parameters for Shi-Tomasi corner detection
          st_feature_params = dict(maxCorners = 2000, qualityLevel = 0.1, minDistance = 5, blockSize = 7)
          # Parameters for Lucas Kanade optical flow field algorithm
          lk_params = dict(winSize = (10,10), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
          cap = cv2.VideoCapture(Video_path)
          surv.lk6(cap, "harris", feature_params, lk_params, 'data/out6_harris_opt.avi', save_vid = True, AM=AM)


      if Do_Part_7:
          # Parameters for Shi-Tomasi corner detection
          feature_params = dict(maxCorners = 2400, qualityLevel = 0.08, minDistance = 2, blockSize = 7, useHarrisDetector=1)
          # Parameters for Harris corner detection
          st_feature_params = dict(maxCorners = 2000, qualityLevel = 0.1, minDistance = 5, blockSize = 7)
          # Parameters for Lucas Kanade optical flow field algorithm
          lk_params = dict(winSize = (10,10), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
          cap = cv2.VideoCapture(Video_path)
          surv.lk7(cap, "harris", feature_params, lk_params, 'data/out7_harris_opt.avi', save_vid = True, AM=AM)
