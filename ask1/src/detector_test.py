import numpy as np
import cv2

def display_corners(corn_arr, rgb_arr, displ=False):
    '''
    displays (in red color) corners from corn_arr in rgb frame in rgb_arr.
    supports cv2.goodFeaturesToTrack sizing : (numofcorners , 1, 2)
    '''
    color = (0,0,255)
    corners_rgb = np.copy(rgb_arr)
    corn_arr_concat = np.concatenate(corn_arr, axis=0).astype(int)
    corn_arr_y = corn_arr_concat[:,0]
    corn_arr_x = corn_arr_concat[:,1]
    #corners_rgb[corn_arr_x, corn_arr_y] = [0,0,255]
    for i in range(corn_arr_x.shape[0]):
        corners_rgb = cv2.circle(corners_rgb, (corn_arr_y[i], corn_arr_x[i]), 1,color, -1 )




    if displ:
        cv2.imshow('dst',corners_rgb)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    return corners_rgb

def detect_corners(frame, params, type="harris", prt=False):
    '''
    Detect interesting points/features of given video's frame
    '''

    '''
    if type=="harris":
        #corner harris returns corner criterion for each pixel
        corners = cv2.cornerHarris(frame, **params)

        #In order to follow cv2.calcOpticalFlowPyrLK interface we
        #have to give Ncorners x 1 x 2 array where the last axis
        #holds x and y coordinates respectively
        #keep only pixels that score high in cornerness criterion
        displ_points = np.where(corners>0.1*corners.max())

        #get x , y axis
        x = np.array([displ_points[0]]).T
        y = np.array([displ_points[1]]).T

        #concatenate to Ncorners x 1 x 2 array
        points = np.concatenate((y,x),axis=1).reshape((x.shape[0],1,2))

    elif type=="st":
        points = cv2.goodFeaturesToTrack(frame, mask=None, **params)
    '''
    if type=="harris":
        points = cv2.goodFeaturesToTrack(frame, mask=None, **params)
    elif type=="st":
        points = cv2.goodFeaturesToTrack(frame, mask=None, **params)

    else:
        raise ValueError('type should one of ["harris", "st"]')

    if prt:
        display_corners(points, frame, type, displ=True)

    return points

if __name__=="__main__":

    '''
    initialize Parameters
    change them to run locally and with
    different feature detectors
    '''
    Video_path = 'data/VIRAT_S_010000_03_000442_000528.mp4'
    # Parameters for Shi-Tomasi corner detection
    st_feature_params = dict(maxCorners = 3000, qualityLevel = 0.08, minDistance = 2, blockSize = 4)
    # Parameters for Harris corner detection
    h_feature_params = dict(blockSize = 6, ksize = 3, k=0.18)

    cap = cv2.VideoCapture(Video_path)
    ret, first_frame = cap.read()
    print(first_frame.shape)
    first_frame = cv2.resize(first_frame, (int(first_frame.shape[1]/2), int(first_frame.shape[0]/2)))

    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    print(first_frame.shape)

    harris_corners = detect_corners(first_frame_gray, h_feature_params, type="harris")
    st_corners = detect_corners(first_frame_gray, st_feature_params, type="st")
    display_corners(harris_corners, first_frame , displ=True)
    display_corners(st_corners, first_frame, displ=True)
