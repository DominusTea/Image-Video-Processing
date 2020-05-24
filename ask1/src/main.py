import numpy as np
import cv2
import detector_test as DT
import draw
import matplotlib.pyplot as plt
import surveillance as surv

if __name__=="__main__":

    Do_Part_4 = True
    Do_Part_5 = False

    Video_path = 'data/VIRAT_S_010000_03_000442_000528.mp4'

    if Do_Part_4:
        #part 4. Appy Lucas-Kanade on video using st or harris corner detection
        display_video = False
        Video_out_path = 'data/out4_video.avi'
        save_vid = False

        # Parameters for Shi-Tomasi corner detection
        st_feature_params = dict(maxCorners = 1000, qualityLevel = 0.08, minDistance = 1, blockSize = 4)
        # Parameters for Harris corner detection
        #h_feature_params = dict(blockSize = 6, ksize = 3, k=0.18)
        h_feature_params = dict(maxCorners = 2000, qualityLevel = 0.1, minDistance = 5, blockSize = 7, useHarrisDetector=1)

        # Parameters for Lucas Kanade optical flow field algorithm
        lk_params = dict(winSize = (15,15), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        #cap = cv2.VideoCapture(Video_path)
        #surv.lk4(cap, "st", st_feature_params, lk_params, 'data/out4_st.avi',save_vid = True)
        cap = cv2.VideoCapture(Video_path)
        surv.lk4(cap, "harris", h_feature_params, lk_params, 'data/out4_harris.avi', save_vid = False)

    #testing Parameters


    if Do_Part_5:
        # Parameters for Shi-Tomasi corner detection
        st_feature_params = dict(maxCorners = 3000, qualityLevel = 0.08, minDistance = 1, blockSize = 4)
        # Parameters for Harris corner detection
        h_feature_params = dict(blockSize = 6, ksize = 3, k=0.18)
        # Parameters for Lucas Kanade optical flow field algorithm
        lk_params = dict(winSize = (15,15), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        cap = cv2.VideoCapture(Video_path)
        surv.lk5(cap, "harris", h_feature_params, lk_params, 'data/out5_st.avi', save_vid = False)




    #part 5. Better following of the points and occasional refreshing of the points we follow
