import numpy as np
import cv2




if __name__=="__main__":

    '''
    initialize Parameters
    change them to run locally and with
    different feature detectors
    '''
    Video_path = 'data/VIRAT_S_010000_03_000442_000528.mp4'
    # Parameters for Shi-Tomasi corner detection
    st_feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
    # Parameters for Harris corner detection
    h_feature_params = dict(blockSize = 6, ksize = 3, k=0.14)



    cap = cv2.VideoCapture(Video_path)
    ret, first_frame = cap.read()
    first_frame = cv2.resize(first_frame, (int(first_frame.shape[1]/2), int(first_frame.shape[0]/2)))

    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    harris_corners = cv2.cornerHarris(first_frame_gray, **h_feature_params)
    st_corners = cv2.goodFeaturesToTrack(first_frame_gray, mask=None, **st_feature_params)

    harris_corners_rgb = np.copy(first_frame)
    st_corners_rgb = np.copy(first_frame)


    print(first_frame.shape)
    print(st_corners.shape)
    st_corners_rgb[st_corners > 0] = [0,0,255]
    harris_corners_rgb[harris_corners>0.01*harris_corners.max()]=[0,0,255]


    cv2.imshow('dst',st_corners_rgb)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
