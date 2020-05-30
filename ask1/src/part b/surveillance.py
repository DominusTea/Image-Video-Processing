import cv2
import detector_test as DT
import draw
import numpy as np
import noise
from skimage.util import random_noise


def lk4(cap, type , detect_params, lk_params, Video_out_path, save_vid = False):

    #Default fps for stored video
    fps = cap.get(cv2.CAP_PROP_FPS)
    #Number of frames
    Nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # read first frame
    ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # Resize frame to half size on both axes
    prev_gray = cv2.resize(prev_gray, (int(first_frame.shape[1]/2), int(first_frame.shape[0]/2)))

    # initilisation of interest points
    prev_points = DT.detect_corners(prev_gray, detect_params, type)

    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(first_frame)
    # resize mask in order to fit image dimensions
    mask = cv2.resize(mask, (int(mask.shape[1]/2), int(mask.shape[0]/2)))

    #Create video writer in order to store the video if save_vid == True
    size = (prev_gray.shape[1],prev_gray.shape[0])

    # video writer object
    out_vid = cv2.VideoWriter(Video_out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    # counter of frames
    frame_index=1

    # process for every video frame
    while (cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < Nframe):

      # Frame reading and processing of size and color space
      ret, frame = cap.read()

      # Every 10 frames mask is set to zero for better display
      if frame_index % 10 ==0:
          mask=np.zeros_like(mask)

      # resize the current frame and converts it to the appropriate color space
      frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # Lucas Kanade algorithm
      next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points.astype(np.float32), None, **lk_params)

      # Selects good feature points for previous position
      good_old = prev_points[status == 1]
      # Selects good feature points for next position
      good_new = next[status == 1]

      # display results
      frame, mask = draw.draw_optfl(good_new, good_old, frame.copy(), mask, only_motion=False)

      # Overlays the optical flow tracks on the original frame
      output = cv2.add(frame, mask)
      # Updates previous frame
      prev_gray = gray.copy()
      # Updates previous good feature points
      prev_points = good_new.reshape(-1, 1, 2)

      # Updates frame index
      frame_index += 1
      # Opens a new window and displays the output frame
      cv2.imshow("sparse optical flow", output)

      # save video
      if save_vid:
          out_vid.write(output)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    cap.release()
    cv2.destroyAllWindows()


def lk5(cap, type , detect_params, lk_params, Video_out_path, save_vid = False):

    #Default fps for stored video
    fps = cap.get(cv2.CAP_PROP_FPS)
    #Number of frames
    Nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read first frame
    ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # resize first frame
    prev_gray = cv2.resize(prev_gray, (int(first_frame.shape[1]/2), int(first_frame.shape[0]/2)))

    # initilisation of interest points
    prev_points = DT.detect_corners(prev_gray, detect_params, type)

    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(first_frame)
    # resize mask in order to fit image dimensions
    mask = cv2.resize(mask, (int(mask.shape[1]/2), int(mask.shape[0]/2)))

    #Create video writer in order to store the video if save_vid == True
    size = (prev_gray.shape[1],prev_gray.shape[0])
    out_vid = cv2.VideoWriter(Video_out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    # counter of frames
    frame_index =  1

    # process for every video frame
    while (cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < Nframe):

      # Frame reading and processing of size and color space
      ret, frame = cap.read()
      frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # Every 10 frames mask is set to zero for better display and interest points are recomputed
      if frame_index % 10 == 0:
          prev_points = DT.detect_corners(prev_gray, detect_params, type)
          mask = np.zeros_like(mask)

      # Lucas Kanade algorithm
      next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points.astype(np.float32), None, **lk_params)

      # Selects good feature points for previous position
      good_old = prev_points[status == 1]
      # Selects good feature points for next position
      good_new = next[status == 1]

      frame, mask = draw.draw_optfl(good_new, good_old, frame.copy(),(mask), only_motion=True)

      # Overlays the optical flow tracks on the original frame
      output = cv2.add(frame, mask)
      # Updates previous frame
      prev_gray = gray.copy()
      # Updates previous good feature points
      prev_points = good_new.reshape(-1, 1, 2)

      # Updates frame index
      frame_index += 1
      # Opens a new window and displays the output frame
      cv2.imshow("sparse optical flow", output)

      # save video
      if save_vid:
          out_vid.write(output)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    cap.release()
    cv2.destroyAllWindows()




def lk6(cap, type , detect_params, lk_params, Video_out_path, save_vid = False, AM = '03116045'):

    # Creates noise seed and amount
    seed = noise.get_seed(AM)
    amount = noise.snpAmount(AM)

    #Default fps for stored video
    fps = cap.get(cv2.CAP_PROP_FPS)
    #Number of frames
    Nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read first frame
    ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # resize first frame
    prev_gray = cv2.resize(prev_gray, (int(first_frame.shape[1]/2), int(first_frame.shape[0]/2)))

    # initilisation of interest points
    prev_points = DT.detect_corners(prev_gray, detect_params, type)

    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(first_frame)
    # resize mask in order to fit image dimensions
    mask = cv2.resize(mask, (int(mask.shape[1]/2), int(mask.shape[0]/2)))

    #Create video writer in order to store the video if save_vid == True
    size = (prev_gray.shape[1],prev_gray.shape[0])
    out_vid = cv2.VideoWriter(Video_out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    # counter of frames
    frame_index =  1

    # process for every video frame
    while (cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < Nframe):

        # Frame reading and processing of size
        ret, frame = cap.read()
        # Resize current frame
        frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
        #insert salt & pepper noise to the frame
        frame[:,:,0] = (255*random_noise(frame[:,:,0], mode='s&p',  seed= seed, amount=amount)).astype('uint8')
        frame[:,:,1] = (255*random_noise(frame[:,:,1], mode='s&p', seed= seed, amount=amount)).astype('uint8')
        frame[:,:,2] = (255*random_noise(frame[:,:,2], mode='s&p', seed= seed, amount=amount)).astype('uint8')

        # conversion of current frame to the appropriate color space
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Every 10 frames mask is set to zero for better display
        if frame_index % 10 == 0:
           mask = np.zeros_like(mask)

        # Every 10 frames find interesting points via a detector
        if frame_index % 10 == 0:
           prev_points = DT.detect_corners(prev_gray, detect_params, type)

        # Lucas Kanade algorithm
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points.astype(np.float32), None, **lk_params)

        # Selects good feature points for previous position
        good_old = prev_points[status == 1]
        # Selects good feature points for next position
        good_new = next[status == 1]

        # display results
        frame, mask = draw.draw_optfl(good_new, good_old, frame.copy(), mask, only_motion=True)

        # Overlays the optical flow tracks on the original frame
        output = cv2.add(frame, mask)
        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev_points = good_new.reshape(-1, 1, 2)

        #Updates frame index
        frame_index += 1
        # Opens a new window and displays the output frame
        cv2.imshow("sparse optical flow", output)

        # save video
        if save_vid:
           out_vid.write(output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    cap.release()
    cv2.destroyAllWindows()

def lk7(cap, type , detect_params, lk_params, Video_out_path, save_vid = False, AM = '03116045'):

    # Creates noise seed and amount
    seed = noise.get_seed(AM)
    amount = noise.snpAmount(AM)

    #Default fps for stored video
    fps = cap.get(cv2.CAP_PROP_FPS)
    #Number of frames
    Nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read first frame
    ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # Resize frame to half
    prev_gray = cv2.resize(prev_gray, (int(first_frame.shape[1]/2), int(first_frame.shape[0]/2)))

    # initilisation of interest points
    prev_points = DT.detect_corners(prev_gray, detect_params, type)

    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(first_frame)
    # resize mask in order to fit image dimensions
    mask = cv2.resize(mask, (int(mask.shape[1]/2), int(mask.shape[0]/2)))

    #Create video writer in order to store the video if save_vid == True
    size = (prev_gray.shape[1],prev_gray.shape[0])
    out_vid = cv2.VideoWriter(Video_out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    #frame counter set to 1
    frame_index =  1
    # process for every video frame
    while (cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < Nframe):

        # Frame reading and processing of size and color space
        ret, frame = cap.read()
        # Resize current frame
        frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))

        #insert salt & pepper noise to the frame
        frame[:,:,0] = (255*random_noise(frame[:,:,0], mode='s&p',  seed=seed, amount=amount)).astype('uint8')
        frame[:,:,1] = (255*random_noise(frame[:,:,1], mode='s&p', seed=seed, amount=amount)).astype('uint8')
        frame[:,:,2] = (255*random_noise(frame[:,:,2], mode='s&p', seed=seed, amount=amount)).astype('uint8')

        # filter the noise using the optimal median filter from Exercise 1
        frame = noise.denoise(frame)
        # conversion of current frame to the appropriate color space
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Every 10 frames mask is set to zero for better display
        if frame_index % 20 == 0:
           prev_points = DT.detect_corners(prev_gray, detect_params, type)
        # Every 10 frames find interesting points via a detector
        if frame_index % 10 ==0:
           mask = np.zeros_like(mask)

        # Lucas Kanade algorithm
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points.astype(np.float32), None, **lk_params)

        # Selects good feature points for previous position
        good_old = prev_points[status == 1]
        # Selects good feature points for next position
        good_new = next[status == 1]
        # diplay results
        frame, mask = draw.draw_optfl(good_new, good_old, frame.copy(), mask, only_motion=True)

        # Overlays the optical flow tracks on the original frame
        output = cv2.add(frame, mask)
        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev_points = good_new.reshape(-1, 1, 2)

        # Updates frame index
        frame_index += 1

        # Opens a new window and displays the output frame
        cv2.imshow("sparse optical flow", output)
        # save video
        if save_vid:
           out_vid.write(output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    cap.release()
    cv2.destroyAllWindows()
