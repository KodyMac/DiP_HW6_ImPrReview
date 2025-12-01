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


def gradients(im, sigma=1.0):
        #do guassian smoothing, sigma is amount of smoothing. float for more accurate
    smoothed = ndimage.guassian_filter(im.astype(float), sigma=sigma)

        #find gradients using sobel, compute x axis first
    Ix = ndimage.sobel(smoothed, axis=1)
    Iy = ndimage.sobel(smoothed, axis=0) #y axis

        #find gradient magnitude
    M = np.sqrt(Ix**2 + Iy**2)

        #direction perp to edge
    theta_rad = np.arctan2(Iy, Ix)
    theta=np.degrees(theta_rad) #convert to degree
    theta[theta<0] += 360 #make sure angle is positive
    #get histogram of gradient orientations
    H, bin_edges = np.histogram(theta.flatten(), bins=360, range=(0,360))
    return Ix, Iy, M, theta, H

def show_gradients(Ix,Iy,M, theta, H, scale_name):
    fig, axis = plt.subplots(2,3, figsize=(15,10))
    fig.suptitle(f'{scale_name} Gradient Analysis', fontsize=16)

    #show gradient in x-direction
    im1 = axis[0,0].imshow(Ix,cmap='gray')
    axis[0,0].set_title('Ix (Gradient in X)')
    axis[0,0].axis('off')
    plt.colorbar(im1, ax=axis[0,0])

    #show gradient in y-direction
    im2 = axis[0,1].imshow(Iy,cmap='gray')
    axis[0,1].set_title('Iy (Gradient in Y)')
    axis[0,1].axis('off')
    plt.colorbar(im2,ax=axis[0,1])

    #show magnitude
    im3 = axis[0,2].imshow(M, cmap='hot')
    axis[0,2].set_title('Magnitude M')
    axis[0,2].axis('off')
    plt.colorbar(im3,ax=axis[0,2])

    #show direction of edges
    im4 = axis[1,0].imshow(theta, cmap='hsv')
    axis[1,0].set_title('Orientation Theta')
    axis[1,0].axis('off')
    plt.colorbar(im4,ax=axis[1,0])

    #show histogram of gradient orientations
    axis[1,1].plot(np.arange(360),H)
    axis[1,1].set_title('Histogram of Gradient Orientations')
    axis[1,1].set_xlabel('Orientation (degrees)')
    axis[1,1].set_ylabel('Frequency')
    axis[1,1].grid(True, alpha=0.3) #somewhat visible grid
    axis[1,2].axis('off')
    plt.tight_layout() #prevent overlap
    return fig


def main():
    im = cv2.imread('NakaHall_2025_small.jpg')
    if im is None:
        print("Error: Could not load image")
        return
    print(f"Image shape: {im.shape}") #print starting dimensions