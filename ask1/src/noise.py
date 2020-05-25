from skimage.morphology import disk, rectangle
import skimage.filters as filters
import numpy as np


def get_seed(AM):
  return int(AM[-1])

def snpAmount(AM):
  '''
  Input: AM (string type)
  Output amount for 'salt and pepper' noise generation
  '''
  digit = AM[-2]
  return float((float(digit)/90) + 0.3)

def denoise(frame, n=2):

    '''
    denoises RGB frame from s&p noise using
    median disk filter with structural elemnt of size = n
    '''
    disk_str = disk(radius=n)
    denoised_frame = np.copy(frame)
    denoised_frame[:,:,0] = filters.rank.median(frame[:,:,0], disk_str )
    denoised_frame[:,:,1] = filters.rank.median(frame[:,:,1], disk_str )
    denoised_frame[:,:,2] = filters.rank.median(frame[:,:,2], disk_str )


    return denoised_frame
