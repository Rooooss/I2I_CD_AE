import cv2
import os
import glob
import numpy as np
import tensorflow as tf
from SSIM_PIL import compare_ssim
from PIL import Image

#real_Images = [cv2.imread(file) for file in glob.glob('C:/Users/JC_PC/Documents/pix2pix-tensorflow/datasets\cityscapes/val/*jpg')]
#val_Images = [cv2.imread(file) for file in glob.glob('C:/Users/JC_PC/Desktop/aefc_concat(node)/*png')]

#real_Images = [Image.open(file) for file in glob.glob('C:/Users/JC_PC/Documents/pix2pix-tensorflow/datasets\cityscapes/val/*jpg')]
#val_Images = [Image.open(file) for file in glob.glob('C:/Users/JC_PC/Desktop/no/*png')]

real_Images = [Image.open(file) for file in glob.glob('C:/Users/Joohong/Desktop/2019/CycleGAN-tensorflow/datasets/facades/testA/*jpg')]
val_Images = [Image.open(file) for file in glob.glob('C:/Users/Joohong/Desktop/2019/CycleGAN-tensorflow/test/*jpg')]

if __name__ == '__main__':

    SSIM_score = 0

    print(len(val_Images))
    for i in range(len(real_Images)):
        r = real_Images[i]
        c = val_Images[i]

        rcSSIM_score = compare_ssim(r, c)
        print(str(rcSSIM_score) + " " + str(i+1))
        SSIM_score += rcSSIM_score

    print(SSIM_score)
    print(SSIM_score/(len(real_Images)))



