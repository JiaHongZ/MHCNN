import os
import cv2
import matplotlib.pyplot as plt
import random
def saltpepper_noise(img, proportion=0.05):
    noise_img = img
    height,width,c =noise_img.shape
    num = int(height*width*proportion)#多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0,width-1)
        h = random.randint(0,height-1)
        if random.randint(0,1) ==0:
            noise_img[h,w,:] =0
        else:
            noise_img[h,w,:] = 255
    return noise_img

image = img = cv2.imread('01.png', 1)  # cv2.IMREAD_GRAYSCALE
print(image.shape)
image = saltpepper_noise(image)
cv2.imwrite('1.png',image)
