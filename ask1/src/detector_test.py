import numpy as np
import cv2

def display_corners(corn_arr, rgb_arr,arr_type="harris", displ=False):
    '''
    displays (in red color) corners from corn_arr in rgb frame in rgb_arr.
    Suppports 2 types of corn_arr input:
        (1) cv2.cornerHarris([xsize, ysize])
        (2) cv2.goodFeaturesToTrack (numofcorners , 1, 2)
    '''

    corners_rgb = np.copy(rgb_arr)

    if arr_type=="harris":
        corners_rgb[corn_arr>0.1*corn_arr.max()]=[0,0,255]
    elif arr_type=="st":
        corn_arr = np.concatenate(st_corners, axis=0).astype(int)
        corn_arr_y = corn_arr[:,0]
        corn_arr_x = corn_arr[:,1]

        corners_rgb[corn_arr_x, corn_arr_y] = [0,0,255]
    else:
        raise ValueError('arr_type corner detection supported is ["harris", "st"]')

    if displ:
        cv2.imshow('dst',corners_rgb)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    return corners_rgb




if __name__=="__main__":

    '''
    initialize Parameters
    change them to run locally and with
    different feature detectors
    '''
    Video_path = 'data/VIRAT_S_010000_03_000442_000528.mp4'
    # Parameters for Shi-Tomasi corner detection
    st_feature_params = dict(maxCorners = 3000, qualityLevel = 0.08, minDistance = 1, blockSize = 4)
    # Parameters for Harris corner detection
    h_feature_params = dict(blockSize = 6, ksize = 3, k=0.18)



    cap = cv2.VideoCapture(Video_path)
    ret, first_frame = cap.read()
    first_frame = cv2.resize(first_frame, (int(first_frame.shape[1]/2), int(first_frame.shape[0]/2)))

    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    harris_corners = cv2.cornerHarris(first_frame_gray, **h_feature_params)
    st_corners = cv2.goodFeaturesToTrack(first_frame_gray, mask=None, **st_feature_params)

    display_corners(harris_corners, first_frame, "harris", displ=True)
    display_corners(st_corners, first_frame, "st", displ=True)
