import numpy as np
import cv2
import matplotlib.pyplot as plt

def display_corners(corn_arr, rgb_arr, displ=False, save=False, path='', title = ''):
    '''
    displays (in red color) corners from corn_arr in rgb frame in rgb_arr.
    supports cv2.goodFeaturesToTrack sizing : (numofcorners , 1, 2)
    '''
    color = (0,0,255)
    corners_rgb = np.copy(rgb_arr)
    #construct corner arr_x/y with size num_of_corners x 2
    corn_arr_concat = np.concatenate(corn_arr, axis=0).astype(int)
    #construct coordinates of corners: corner arr_x/y with size: num_of_corners x 1
    corn_arr_y = corn_arr_concat[:,0]
    corn_arr_x = corn_arr_concat[:,1]

    #display a red circle over every detected corner
    for i in range(corn_arr_x.shape[0]):
        corners_rgb = cv2.circle(corners_rgb, (corn_arr_y[i], corn_arr_x[i]), 1,color, -1 )


    #display detected corners
    if displ:
        cv2.imshow(title, corners_rgb)
        fig = plt.figure()
        plt.imshow(cv2.cvtColor(corners_rgb, cv2.COLOR_BGR2RGB))
        plt.title(title, color='b')
        # save image
        if save:
            plt.savefig(path,bbox_inches='tight', dpi=600)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    return corners_rgb

def detect_corners(frame, params, type="harris", prt=False):
    '''
    Detect interesting points/features of given video's frame.
    Uses a little bit clunky interface due to changes in
    the way interesting points are detected.
    Only params determine which detector will be used.
    '''
    # harris method
    if type=="harris":
        points = cv2.goodFeaturesToTrack(frame, mask=None, **params)
    # Shai-Tomasi method
    elif type=="st":
        points = cv2.goodFeaturesToTrack(frame, mask=None, **params)

    else:
        raise ValueError('type should one of ["harris", "st"]')

    # display interest points
    if prt:
        display_corners(points, frame,  displ=True)

    return points

if __name__=="__main__":

    '''
    The following code is used to test different kind of detectors.
    (Part 3 of Lab exercise)
    initialize Parameters
    change them to run locally and with
    different feature detectors
    '''
    Video_path = 'data/VIRAT_S_010000_03_000442_000528.mp4'

    # Parameters for Shi-Tomasi corner detection
    st_feature_params = dict(maxCorners = 2000, qualityLevel = 0.1, minDistance = 2, blockSize = 7)    # Parameters for Harris corner detection
    h_feature_params = dict(maxCorners = 2400, qualityLevel = 0.08, minDistance = 2, blockSize = 7, useHarrisDetector=1)

    # video capture object
    cap = cv2.VideoCapture(Video_path)
    # reading of first frame
    ret, first_frame = cap.read()
    # resize of first frame
    first_frame = cv2.resize(first_frame, (int(first_frame.shape[1]/2), int(first_frame.shape[0]/2)))
    # conversion to appropriate color space
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # interest points detection using the two methods
    harris_corners = detect_corners(first_frame_gray, h_feature_params, type="harris")
    st_corners = detect_corners(first_frame_gray, st_feature_params, type="st")

    # display results
    display_corners(harris_corners, first_frame , displ=True, save=True, path="data/detector_harris_opt.png", \
                    title='Harris detector using maxCorners = ' + str(h_feature_params['maxCorners']) + ' , qualityLevel = ' + \
                    str(h_feature_params['qualityLevel']) + ', \n minDistance = ' + str(h_feature_params['minDistance'])\
                + ', blockSize = ' + str(h_feature_params['blockSize']))

    display_corners(st_corners, first_frame, displ=True, save=True, path="data/detector_st_opt_test.png", \
                    title='Shai-Tomasi detector using maxCorners = ' + str(st_feature_params['maxCorners']) + ' , qualityLevel = ' + \
                    str(st_feature_params['qualityLevel']) + ', \n minDistance = ' + str(st_feature_params['minDistance'])\
                + ', blockSize = ' + str(st_feature_params['blockSize']))
