# TASK -- CREATE YOUR OWN CANNY EDGE DETECTOR

# IMPORT PACKAGES INCLUDING math,skimage,numpy,scipy and matplotlib
import math as m
from skimage import feature as f
from skimage import data
from skimage.filters import *
from skimage.measure import compare_ssim as ssim
from skimage.color import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from skimage import io

# Function mse , is used to find mean squared error between the values of two output images
# MSE is called by other functions below
# parameters of the function include the two images
def mse(imageA, imageB):
    # error is calculated using mathematical formula for mse
    # error = ((yi)^2-(y0)^2)/N
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# Function PSNR is used to find peak signal to noise ratio between two images
# Note that its important that the two images are in same format, i.e
# either both images are coloured or both are greyscale
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
    # MSE is zero means no noise is present in the signal .
    # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    # calculate PSNR using mathematical formula
    # PSNR= 20*m*log10(MAX_PIXEL_VALUE/SQUARE_ROOT(MSE))
    psnr = 20 * m.log10(max_pixel / m.sqrt(mse))
    return psnr

# Function compare_images, is used to find mean squared error (MSE) and
# Structural Similarity Index Metric (SSIM) and display with two comapred images

def compare_image(imageA, imageB, title):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    p=PSNR(imageA,imageB)
    fig = plt.figure(title)
    plt.suptitle("PSNR: %.2f, SSIM: %.2f" % (p, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap='gray')
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap='gray')
    plt.axis("off")
    plt.show()
    return s



# Function myCannyEdgeDetector() is used to derive thin edges of the given Image.

def myCannyEdgeDetector(image, lowThreshold, highThreshold):
    # Step 1: Convert to Grayscale in order to get a 2D array
    image = rgb2gray(image)
    # Apply Gaussian Filter to smoothe the image, here we use sigma = 0.6


    def Gaussian_mask(m, n, sigma):
        gaussian_b = np.zeros((m, n))
        m = m // 2
        n = n // 2
        for x in range(-m, m + 1):
            for y in range(-n, n + 1):
                x1 = sigma * (2 * np.pi) ** 2
                x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                gaussian_b[x + m, y + n] = (1 / x1) * x2
        return gaussian_b

    def corr(img, mask):
        row, col = img.shape
        m, n = mask.shape
        new = np.zeros((row + m - 1, col + n - 1))
        m = m // 2
        n = n // 2
        filtered_img = np.zeros(img.shape)
        new[m:new.shape[0] - m, n:new.shape[1] - n] = img
        for i in range(m, new.shape[1] - m):
            for j in range(n, new.shape[1] - n):
                temp = new[i - m:i + m + 1, j - m:j + m + 1]
                result = temp * mask
                filtered_img[i - m, j - n] = result.sum()
        return filtered_img

    gaussian_mask = Gaussian_mask(5, 5,2)
    inputImgGaus = corr(image, gaussian_mask)

    # Step2 Find Magnitude and Orientation of Gradient
    # Computing Ix and Iy
    Ix = convolve(inputImgGaus,[[-1,0,1],[-2,0,2],[-1,0,1]])
    Iy = convolve(inputImgGaus,[[1,2,1],[0,0,0],[-1,-2,-1]])
    # Calculating Magnitude and Degree
    I = np.power(np.power(Ix, 2.0) + np.power(Iy, 2.0), 0.5)
    theta = np.degrees(np.arctan2(Iy, Ix))

    # Gradient intensity matrix
    Mag = np.hypot(Ix, Iy)

    # Step 3- Calculate NMS or Non Maximum Suppression
    # We traverse over the image and check orientation and magnitude for each
    # range of degrees for 0 to 360


    def NMS(M, angle, Gradient_x, Gradient_y):
        NMS = np.zeros(M.shape)
        for i in range(1, int(M.shape[0] - 1)):
            for j in range(1, int(M.shape[1] - 1)):
                if ((angle[i, j] >= 0 and angle[i, j] <= 45) or (angle[i, j] >= -180 and angle[i, j] <= -135)):
                    trav_1 = np.array([M[i, j], M[i + 1, j + 1]])
                    trav_2 = np.array([M[i, j - 1], M[i + 1, j - 1]])
                    est = np.absolute(Gradient_y[i, j] / M[i, j])
                    if ((M[i, j] >= ((trav_1[1] - trav_1[0]) * est + trav_1[0])) & (
                    (M[i, j] >= ((trav_1[1] - trav_1[0]) * est + trav_1[0])))):
                        NMS[i, j] = M[i, j]
                    else:
                        NMS[i, j] = 0

        for i in range(1, int(M.shape[0] - 1)):
            for j in range(1, int(M.shape[1] - 1)):
                if ((angle[i, j] >= 45 and angle[i, j] <= 90) or (angle[i, j] >= -135 and angle[i, j] <= -90)):
                    trav_1 = np.array([M[i + 1, j], M[i + 1, j + 1]])
                    trav_2 = np.array([M[i, j - 1], M[i - 1, j - 1]])
                    est = np.absolute(Gradient_x[i, j] / M[i, j])
                    if ((M[i, j] >= ((trav_1[1] - trav_1[0]) * est + trav_1[0])) & (
                    (M[i, j] >= ((trav_1[1] - trav_1[0]) * est + trav_1[0])))):
                        NMS[i, j] = M[i, j]
                    else:
                        NMS[i, j] = 0

        for i in range(1, int(M.shape[0] - 1)):
            for j in range(1, int(M.shape[1] - 1)):
                if ((angle[i, j] >= 90 and angle[i, j] <= 135) or (angle[i, j] >= -90 and angle[i, j] <= -45)):
                    trav_1 = np.array([M[i + 1, j], M[i + 1, j - 1]])
                    trav_2 = np.array([M[i - 1, j - 1], M[i - 1, j + 1]])
                    est = np.absolute(Gradient_x[i, j] / M[i, j])
                    if ((M[i, j] >= ((trav_1[1] - trav_1[0]) * est + trav_1[0])) & (
                    (M[i, j] >= ((trav_1[1] - trav_1[0]) * est + trav_1[0])))):
                        NMS[i, j] = M[i, j]
                    else:
                        NMS[i, j] = 0

        for i in range(1, int(M.shape[0] - 1)):
            for j in range(1, int(M.shape[1] - 1)):
                if ((angle[i, j] >= 135 and angle[i, j] <= 180) or (angle[i, j] >= -45 and angle[i, j] <= 0)):
                    trav_1 = np.array([M[i, j - 1], M[i + 1, j - 1]])
                    trav_2 = np.array([M[i, j + 1], M[i - 1, j + 1]])
                    est = np.absolute(Gradient_y[i, j] / M[i, j])
                    if ((M[i, j] >= ((trav_1[1] - trav_1[0]) * est + trav_1[0])) & (
                    (M[i, j] >= ((trav_1[1] - trav_1[0]) * est + trav_1[0])))):
                        NMS[i, j] = M[i, j]
                    else:
                        NMS[i, j] = 0
        return NMS

    #Step4-Join the Strong and Weak Edges
    def DoThreshHyst(img):
        highThresholdRatio = 0.30
        lowThresholdRatio = 0.10
        GSup = np.copy(img)
        h = int(GSup.shape[0])
        w = int(GSup.shape[1])
        highThreshold = np.max(GSup) * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio
        print("High Threshold :",highThreshold )
        print("Low Threshold :",lowThreshold)

        # The while loop is used so that the loop will keep executing till the number of strong edges do not change,
        # i.e all weak edges connected to strong edges have been found
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if (GSup[i, j] > highThreshold):
                    GSup[i, j] = 1
                elif (GSup[i, j] < lowThreshold):
                    GSup[i, j] = 0
                else:
                    if ((GSup[i - 1, j - 1] > highThreshold) or
                            (GSup[i - 1, j] > highThreshold) or
                            (GSup[i - 1, j + 1] > highThreshold) or
                            (GSup[i, j - 1] > highThreshold) or
                            (GSup[i, j + 1] > highThreshold) or
                            (GSup[i + 1, j - 1] > highThreshold) or
                            (GSup[i + 1, j] > highThreshold) or
                            (GSup[i + 1, j + 1] > highThreshold)):
                        GSup[i, j] = 1

            return GSup


    New_image = NMS(Mag, theta, Ix, Iy)

    Final_img = DoThreshHyst(New_image)

    return Final_img

# Take input as image
inputImg = data.astronaut()
# Convert to gray scale
grey_img=rgb2gray(inputImg)
# Take output of My_canny_edge_detector
my_output = myCannyEdgeDetector(inputImg, 0.5, 0.15)
# Take output of inbuilt canny edge detector
standard_output=f.corner_harris(grey_img,k=1,sigma=2)
# Calculating MSE and PSNR
MSE=mse(my_output,standard_output)
error=PSNR(my_output,standard_output)
# Display the Final Result
SSIM=compare_image(my_output,standard_output,"COMPARISON BETWEEN TWO OUTPUTS")
print("MSE :",MSE,"PSNR :",error,"SSIM :",SSIM)

