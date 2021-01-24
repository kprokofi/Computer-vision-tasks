"""by the given image of the billiard balls need to count them and approximately find their radius. Then compute the variance for the found radiuses."""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def count_balls(img, method="erode"):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    if method=="erode":
        morphed_img = cv.erode(img, kernel)
    else:
        assert method == "opening"
        morphed_img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=2)

    cv.imshow('After erosing', cv.resize(morphed_img, (720, 560)))
    contours, hierarchy = cv.findContours(morphed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    count_balls = 0
    area_store = []
    non_noisy_cntrs = []
    for i, c in enumerate(contours):
        # compute cntour area
        area = cv.contourArea(c)
        # get rid of noise
        if area > 3000:
            non_noisy_cntrs.append(c)
            # if the area more than the empiric threshold
            # that means the balls weren't separated.
            if area > 15000:
                count_balls += 2
                area_store.append(area / 2)
            else:
                count_balls += 1
                area_store.append(area)
    rads = []
    for ar in area_store:
        # compure radius (pi*r**2 = S)
        rads.append(np.sqrt(ar/np.pi)) 
    
    return count_balls, rads, non_noisy_cntrs

def main():
    init_img = cv.imread("..\image_example\whiteballssample.jpg")
    gray_scale_img = cv.cvtColor(init_img, cv.COLOR_BGR2GRAY)
    img1 = cv.equalizeHist(gray_scale_img)
    ret, img2 = cv.threshold(img1,240,255, cv.THRESH_BINARY)

    blur = cv.GaussianBlur(gray_scale_img,(5,5),0)
    ret3,th3 = cv.threshold(gray_scale_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    th2 = cv.adaptiveThreshold(gray_scale_img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                cv.THRESH_BINARY,33,2)

    titles = ['Original Image', 'Binary Thresholding (trsh = 240)',
                'Adaptive Mean Thresholding', 'Otsu']
    images = [gray_scale_img, img2, th2, th3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    for img, name in zip([th3, img2], ['Otsu', 'Binary']):
        num_balls, rads, non_noisy_cntrs = count_balls(img, method="erode")

        print("ball count on the image: {}, method: {}".format(num_balls, name))
        print("average radius of the balls: {}, method: {}".format(np.mean(rads), name))
        print("variance of the radius: {}, method: {}\n".format(np.var(rads), name))

        contoured_image = gray_scale_img.copy()
        cv.drawContours(contoured_image, non_noisy_cntrs, -1, (0,255,0), 3)
        cv.imshow('contours drawing, method: {}'.format(name), cv.resize(contoured_image, (720, 560)))
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()