import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_optfl(good_new, good_old,frame, mask , only_motion=False):
    '''
    Draws the optical flow tracks on given mask using old and new detected interesting points.
    If only_motion=False (default) draw a line on mask for every movement by an interesting point.
    In this case, also update frame with a circle over detected interesting points.

    If only_motion=True, draw a line on mask only for interesting points that move by a sum of
    at least 2 pixels in both axes
    '''
    color=(0,0,255) #this color is red: the color of passion!
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # a, b = coordinates of new point
        a, b = new.ravel()
        # c, d = coordinates of old point
        c, d = old.ravel()

        # Draws line between new and old position with red color and 2 thickness in case we choose not only motion display
        if not(only_motion):
            mask = cv2.line(mask, (a, b), (c, d), color, 2)

        # Draws line between new and old position with red color and 2 thickness in case we choose only motin display
        # and the movement of corners is at least 2 pixels
        elif (abs(a-c) + abs(b-d)) > 1:
            mask = cv2.line(mask, (a,b), (c,d), color, 2)

        # Draws filled circle (thickness of -1) at new position with red color and radius of 1
        if not(only_motion):
            frame = cv2.circle(frame, (a, b), 1, color, -1)

    return frame, mask
