import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

#translate image by (tx, ty) pixels
def Translate(im, tx, ty):
    h,w = im.shape[:2]
    #make the special matrix for translations
    translate_matrix = np.float32([[1,0,tx],[0,1,ty]])
    translated = cv2.warpAffine(im, translate_matrix, (w,h))
    return translated

def CropScale(im, x1, y1, x2, y2, s):
    #make sure values are correctly ordered
    x_min, x_max = min(x1,x2), max(x1,x2)
    y_min, y_max = min(y1,y2), max(y1,y2)

    #crop region by slicing array
    cropped = im[y_min:y_max, x_min:x_max]

    #find new dimensions from scale
    new_width = int(cropped.shape[1] * s)
    new_height = int(cropped.shape[0]*s)

    #interlinear gives smoother scaling
    scaled = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return scaled

def Vertical_Flip(im):
    return cv2.flip(im,1)

def Horizontal_Flip(im):
    return cv2.flip(im,0)

def Rotate(im, angle):
    h,w = im.shape[:2]

    center = (w//2, h//2)

    #rotation needs center point, the angle, and the scale factor
    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(im, rotation_mat, (w,h)) #rotate image using matrix, keep same size as input
    return rotated

def Fill(im, x1,y1,x2,y2,val):
    result = im.copy() #don't mess with original

    #make sure correct order
    x_min,x_max = min(x1,x2), max(x1,x2)
    y_min,y_max = min(y1,y2), max(y1,y2)

    #fill region with value, slicing array to get region
    result[y_min:y_max, x_min:x_max] = val
    return result