import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

def read_image(path: str):
    image = cv.imread(path)
    image_YUV = cv.cvtColor(cv.resize(image, (1280,720), cv.INTER_CUBIC), cv.COLOR_BGR2YUV)
    # cv.imshow('corners_window0', image_YUV)
    return image, image_YUV

def step_3_8(img, k_dist=15, rad=5, cor_thrsh = 0.05, use_dilate=True, white=False):
    equ = cv.equalizeHist(img)
    edges = cv.Canny(equ,100,200)
    cv.imshow('Edges', cv.resize(edges, (1280,720)))
    
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04

    # Detecting corners
    corners = cv.cornerHarris(equ, blockSize, apertureSize, k)

    # Threshold for an optimal value
    mask = corners > cor_thrsh*corners.max()
    img[corners > cor_thrsh*corners.max()]=[0]

    # Drawing a circle around corners by given threshold
    for i in range(corners.shape[0]):
        for j in range(corners.shape[1]):
            if mask[i,j]:
                cv.circle(edges, (j,i), rad, (255), -1)

    # Showing the result
    if use_dilate:
        edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    cv.imshow('corners', edges)

    dst = cv.distanceTransform(edges, cv.DIST_L2, 3)
    cv.normalize(dst, dst, 0, 1., cv.NORM_MINMAX)
    cv.imshow('Distance Transform Image', cv.resize(dst, (1280,720)))
    integral_img = cv.integral(edges)
    output = adaptive_kernel(equ, integral_img, dst, k_dist, white)
    return output

@njit
def adaptive_kernel(orig_img, integral_img, dst_map, k, white=False):
    output = np.empty_like(orig_img)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            ker_size = int(k*dst_map[i][j])
            if ker_size == 0:
                output[i][j] = orig_img[i][j]
                continue
            # clip if coordinates are out of the range
            x1, y1 = max(i - ker_size, 0), max(j - ker_size, 0)
            x2, y2 = min(i + ker_size, output.shape[0]-1), min(j + ker_size, output.shape[1]-1)
            # sum kernel
            left = (integral_img[x2][y1] + integral_img[x1][y2])
            right = (integral_img[x2][y2] + integral_img[x1][y1])
            if white:
                I_sum = right - left
            else:
                I_sum = left - right
            output[i][j] = I_sum/(2*(ker_size)**2)
    return output

def main():
    _, img = read_image("..\image_example\house.jpg")
    output = step_3_8(img[:,:,0], k_dist=20, rad=3, use_dilate=True, white=False)
    output = np.clip(output,0,255)
    img[:,:,0] = output
    img = cv.cvtColor(img, cv.COLOR_YUV2BGR)
    cv.imwrite('final_img.jpg', img)
    cv.waitKey()

if __name__=="__main__":
    main()