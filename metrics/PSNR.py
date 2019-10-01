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

if __name__ == '__main__':

    psnr = 0

    real_Images = [Image.open(file) for file in
                   glob.glob('C:/Users/Joohong/Desktop/2019/CycleGAN-tensorflow/datasets/facades/testA/*jpg')]
    val_Images = [Image.open(file) for file in glob.glob(
        'C:/Users/Joohong/Desktop/2019/CycleGAN-tensorflow/test/*jpg')]

    sess = tf.InteractiveSession()
    for i in range(len(real_Images)):
        im1 = tf.image.convert_image_dtype(real_Images[i], tf.float32)
        im2 = tf.image.convert_image_dtype(val_Images[i], tf.float32)

        psnr += tf.image.psnr(im1, im2, max_val=1.0)

    print("psnr의 총합은 : ", str(psnr.eval()))
    print("psnr의 평균값은 : ", str(psnr.eval()/(len(real_Images))))
    sess.close()



