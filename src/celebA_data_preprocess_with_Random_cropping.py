import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
from PIL import Image
import scipy.ndimage as ndimage
import random
import numpy as np
# root path depends on your computer
root = '/home/davidk/Downloads/celeba_dataset/Img/img_align_celeba/'
save_root = '/home/davidk/Downloads/celeba_dataset/random_cropping/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

rand_width= 16
d = rand_width-1
for i in range(len(img_list)):

    img = Image.open(root + img_list[i])
    img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
    seed_rand_x = np.random.randint(0, (resize_size/rand_width)+1, size=1)[0]
    seed_rand_y = np.random.randint(0, (resize_size/rand_width)+1, size=1)[0]
    img_to_np_array= np.array(img)
    img_to_np_array[rand_width*seed_rand_y:rand_width*seed_rand_y+(d), rand_width*seed_rand_x:rand_width*seed_rand_x+(d), :] = 200
    img = Image.fromarray(np.uint8(img_to_np_array))
    img.save(save_root + 'celebA/' + img_list[i])
    if i == 300:
        break
    if (i % 1000) == 0:
        if i==500 :
            break
        print('%d images complete' % i)