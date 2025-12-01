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

    print("\n1. Apply translation (tx=300, ty=200)")
    translated = Translate(im, 300,200)
    cv2.imwrite('output_translated.jpg', translated)
    print(" Saved output")

    print("\n2. Apply crop and scale (x1=1, y1=1600, x2=600, y2=1200, s=0.5)")
    cropped_scaled = CropScale(im, 1, 1600, 600, 1200, 0.5)
    cv2.imwrite('output_cropscale.jpg', cropped_scaled)
    print(" Saved output")

    print("\n3. Apply vertical flip")
    v_flip = Vertical_Flip(im)
    cv2.imwrite('output_vflip.jpg', v_flip)
    print(" Saved output")

    print("\n4. Apply horizontal flip")
    h_flip = Horizontal_Flip(im)
    cv2.imwrite('output_hflip.jpg', h_flip)
    print(" Saved output")

    print("\n5. Apply rotation (angle = 60)")
    rotated = Rotate(im, 60)
    cv2.imwrite('output_rotated.jpg', rotated)
    print(" Saved output")

    print("\n6. Apply fill (x1=500, y1=1000, x2=1000, y2=800, val=150)")
    filled = Fill(im, 500, 1000, 1000, 800, 150)
    cv2.imwrite('output_filled.jpg', filled)
    print(" Saved output")


    print("\nExtracting green channel")
    green_chan = im[:,:,1]

    print("\nComputing gradient at small scale")
    Ix_s, Iy_s, M_s, theta_s, H_s = gradients(green_chan, sigma=1)

    cv2.imwrite('output_Ix_small.jpg', cv2.normalize(Ix_s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite('output_Iy_small.jpg', cv2.normalize(Iy_s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite('output_M_small.jpg', cv2.normalize(M_s, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    #visualize small scale results in one thing
    fig_small = show_gradients(Ix_s, Iy_s, M_s, theta_s, H_s, "Small Scale (theta=1.0)")

    #save figure
    fig_small.savefig('output_gradients_small_scale.png', dpi=150, bbox_inches='tight')
    print(" Saved output")

    #large scale
    print("\nComputing gradients at large scale (sigma=3.0)")
    Ix_l, Iy_l, M_l, theta_l, H_l = gradients(green_chan, sigma=3.0)

    #save images
    cv2.imwrite('output_Ix_large.jpg', cv2.normalize(Ix_l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite('output_Iy_large.jpg', cv2.normalize(Iy_l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite('output_M_large.jpg', cv2.normalize(M_l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    #visualize
    fig_large = show_gradients(Ix_l, Iy_l, M_l, theta_l, H_l, "Large Scale (theta=3.0)")
    fig_large.savefig('output_gradients_large_scale.png', dpi=150, bbox_inches='tight')
    print(" Saved output")

    print("Completed! All output files are saved")

    plt.show() #show matplotlib figures

if __name__ == "__main__":
    main()