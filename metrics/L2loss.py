import cv2
import os
import glob
import numpy as np

L1_real_images = [cv2.imread(file) for file in glob.glob('C:/Users/Joohong/Desktop/2019/CycleGAN-tensorflow/datasets/facades/testA/*jpg')]
L1_val_images = [cv2.imread(file) for file in glob.glob('C:/Users/Joohong/Desktop/2019/CycleGAN-tensorflow/test/*jpg')]


def mse(imageA, imageB):

    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension

    err = np.sum((imageA.astype("float") - imageB.astype("float"))**2)
    err /= float(imageA.shape[0] * imageA.shape[1] * 3)

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

if __name__ == '__main__':

    L2_Loss = 0

    for i in range(len(L1_real_images)):

        # grayscaling
        #real_gray = cv2.cvtColor(L1_real_images[i], cv2.COLOR_BGR2GRAY)
        #val_gray = cv2.cvtColor(L1_val_images[i], cv2.COLOR_BGR2GRAY)
        #L2_Loss += mse(real_gray, val_gray)

        #no grayscaling
        L2_Loss += mse(L1_real_images[i], L1_val_images[i])

    print(L2_Loss/(len(L1_real_images)))

