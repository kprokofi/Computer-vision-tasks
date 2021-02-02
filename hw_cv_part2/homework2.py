import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def read_image(path: str):
    image = cv.imread(path)
    image_YUV = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    # cv.imshow('corners_window0', image_YUV)
    return image, image_YUV

def step_3_8(img):
    thresh = 100
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = img
    equ = cv.equalizeHist(gray)
    edges = cv.Canny(equ,100,200)
    # cv.imshow('corners_window1', edges)
    
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04

    # Detecting corners
    print(img.shape)
    dst = cv.cornerHarris(equ, blockSize, apertureSize, k)
    dst = cv.dilate(dst,None)
    print(dst.shape)

    # Threshold for an optimal value, it may vary depending on the image.
    mask = dst>0.05*dst.max()
    print(dst)
    # img[dst>0.01*dst.max()] = 255
    # img[dst>0.01*dst.max()]=[0,0,255]
    img[dst>0.01*dst.max()]=[0]
    cv.imwrite('corners0.jpg', img)

    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if mask[i,j]:
                cv.circle(edges, (j,i), 7, (255), -1)


    # Showing the result
    cv.imwrite('corners.jpg', edges)
    edges = cv.dilate(edges,None)

    dst = cv.distanceTransform(edges, cv.DIST_L2, 3)
    cv.imshow('Distance Transform Image', cv.resize(dst, (1280,720)))
    integral_img = cv.integral(equ)
    output = adaptive_kernel(equ, integral_img, dst)
    return output

def adaptive_kernel(orig_img, integral_img, dst_map, k=3):
    output = np.empty_like(orig_img)
    print(output.shape[0], output.shape[1])
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            ker_size = int(k*dst_map[i][j])

            # clip if coordinates are out of the range
            x1, y1 = max(i - ker_size, 0), max(j - ker_size, 0)
            x2, y2 = min(i + ker_size, output.shape[0]-1), min(j + ker_size, output.shape[1]-1)
            I_sum = (integral_img[x2][y2] + integral_img[x1][y1]) - (integral_img[x2][y1] + integral_img[x1][y2])
            print(I_sum, ker_size)
            output[i][j] = I_sum/(2*(ker_size + 1e-6)**2)
    return output

def main():
    # img0, img = read_image("C:\\Users\\kirill\\OpenCV\\image_example\\cats.jpg")
    img = cv.imread("C:\\Users\\kirill\\OpenCV\\hw_cv_part2\\final.jpg")
    # output = step_3_8(img[:,:,0])
    # img[:,:,0] = np.clip(output,0,255)
    img = cv.cvtColor(img, cv.COLOR_YUV2BGR)
    cv.imwrite('final.jpg', img)
    # step_3_8(img0)
    cv.waitKey()

if __name__=="__main__":
    main()