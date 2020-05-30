import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_optfl(good_new, good_old,frame, mask , only_motion=False):
 # Draws the optical flow tracks
    color=(0,0,255)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # a, b = coordinates of new point
        a, b = new.ravel()
        # c, d = coordinates of old point
        c, d = old.ravel()

        # Draws line between new and old position with red color and 2 thickness

        if not(only_motion):
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
        #elif abs(a-c)>1 or abs(b-d)>1:
        elif (abs(a-c) + abs(b-d)) > 1:
            #mask = cv2.line(mask-prev_10_mask*(((mask > 0).astype(int)-(prev_10_mask > 0).astype(int)) > 0) , (a,b), (c,d), color, 2)

            mask = cv2.line(mask, (a,b), (c,d), color, 2)
            #mask = mask-prev_10_mask
            #plt.imshow(mask)
            #plt.show()

            #plt.imshow(mask)
            #plt.show()
        # Draws filled circle (thickness of -1) at new position with red color and radius of 3
        if not(only_motion):
            frame = cv2.circle(frame, (a, b), 1, color, -1)

        '''
        if abs(a-c)>3 or abs(b-d)>3:
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
        frame=cv2.circle(frame, (a,b), 1, color,-1)
        '''
    return frame, mask

def update(mask, time_mask):
    '''
    returns updated masked, where old detected corners are erased
    '''
    mask[time_mask==10] = 0
    return mask
