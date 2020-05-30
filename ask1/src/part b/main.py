import numpy as np
import cv2
import detector_test as DT
import draw
import matplotlib.pyplot as plt
import surveillance as surv

if __name__=="__main__":

    '''
    The Do_Part variables below control which part of the exercise will be run.
    Simply set one (or many) to True in order to run a specific part of the exercise
    Do not forget to change Video_path variable to your local directory.
    lk_i take as arguments a videocapture object, the detectors parameter,
    the lucas-kanade parameters, the out directory, and a boolean which controls if the video is saved
    '''
    Do_Part_4 = False
    Do_Part_5 = False
    Do_Part_6 = False
    Do_Part_7 = True
    AM =  '03116045' #Danae's AM

    Video_path = 'data/VIRAT_S_010000_03_000442_000528.mp4'

    #part 4. Appy Lucas-Kanade on video using st or harris corner detection

    if Do_Part_4:

        Video_out_path = 'data/out4_video.avi'
        save_vid = False

        # Parameters for Shi-Tomasi corner detection
        st_feature_params = dict(maxCorners = 2000, qualityLevel = 0.1, minDistance = 2, blockSize = 7)
        # Parameters for Harris corner detection
        h_feature_params = dict(maxCorners = 2400, qualityLevel = 0.08, minDistance = 2, blockSize = 7, useHarrisDetector=1)

        # Parameters for Lucas Kanade optical flow field algorithm
        lk_params = dict(winSize = (10,10), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # video capture object
        cap = cv2.VideoCapture(Video_path)
        # tracking of interest points
        surv.lk4(cap, "st", st_feature_params, lk_params, 'data/out4_st_opt.avi', save_vid = False)



    #part 5. Better following of the points and occasional refreshing of the points we follow

    if Do_Part_5:

         # Parameters for Harris corner detection
         feature_params = dict(maxCorners = 2400, qualityLevel = 0.08, minDistance = 2, blockSize = 7, useHarrisDetector=1)
         # Parameters for Shi-Tomasi corner detection
         st_feature_params = dict(maxCorners = 2000, qualityLevel = 0.1, minDistance = 2, blockSize = 7)

         # Parameters for Lucas Kanade optical flow field algorithm
         lk_params = dict(winSize = (10,10), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

         # video capture object
         cap = cv2.VideoCapture(Video_path)
         # tracking of interest points
         surv.lk5(cap, "harris", feature_params, lk_params, 'data/out5_harris_opt.avi', save_vid = False)


    #part 6. tracking of interest points after noise insertion

    if Do_Part_6:

         # Parameters for Harris corner detection
         feature_params = dict(maxCorners = 2400, qualityLevel = 0.08, minDistance = 2, blockSize = 7, useHarrisDetector=1)
         # Parameters for Shi-Tomasi corner detection
         st_feature_params = dict(maxCorners = 2000, qualityLevel = 0.1, minDistance = 5, blockSize = 7)

         # Parameters for Lucas Kanade optical flow field algorithm
         lk_params = dict(winSize = (10,10), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

         # video capture object
         cap = cv2.VideoCapture(Video_path)
         # tracking of interest points
         surv.lk6(cap, "harris", feature_params, lk_params, 'data/out6_harris_opt.avi', save_vid = False, AM=AM)

    #part 7. tracking of interest points after noise insertion and noise filtering using median filter found in First exercise

    if Do_Part_7:

         # Parameters for Shi-Tomasi corner detection
         feature_params = dict(maxCorners = 2400, qualityLevel = 0.08, minDistance = 2, blockSize = 7, useHarrisDetector=1)
         # Parameters for Harris corner detection
         st_feature_params = dict(maxCorners = 2000, qualityLevel = 0.1, minDistance = 5, blockSize = 7)

         # Parameters for Lucas Kanade optical flow field algorithm
         lk_params = dict(winSize = (10,10), maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

         # video capture object
         cap = cv2.VideoCapture(Video_path)
         # tracking of interest points
         surv.lk7(cap, "st", st_feature_params, lk_params, 'data/out7_st_opt_3.avi', save_vid = False, AM=AM)
