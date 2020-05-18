import numpy as np
import cv2





if __name__=="__main__":

    Video_path = 'data/VIRAT_S_010000_03_000442_000528.mp4'
    display_video = False

    cap = cv2.VideoCapture(Video_path)
    while (cap.isOpened()):
      ret, frame = cap.read()
      frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
      if display_video:
          cv2.imshow("Surveillance video", frame)

      frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)








      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    cap.release()
    cv2.destroyAllWindows()
