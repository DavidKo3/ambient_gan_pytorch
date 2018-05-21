import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
from PIL import Image
import scipy.ndimage as ndimage
import random
import numpy as np
# root path depends on your computer
root = '/home/davidk/Downloads/celeba_dataset/Img/img_align_celeba/'
save_root = '/home/davidk/Downloads/celeba_dataset/gaussian_noise_celeba/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):

    img = Image.open(root + img_list[i])
    img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
    blurred_img = ndimage.gaussian_filter(img, sigma=(1, 1, 0), order=0)
    blurred_img_add_noise = blurred_img + np.random.normal(0, 0.2, blurred_img.shape)
    blurred_img_add_noise_pil = Image.fromarray(np.uint8(blurred_img_add_noise))
    blurred_img_add_noise_pil.save(save_root + 'celebA/' + img_list[i])
    if (i % 1000) == 0:
        print('%d images complete' % i)
