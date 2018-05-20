import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
from PIL import Image
# root path depends on your computer
root = '/home/davidk/Downloads/celeba_dataset/Img/img_align_celeba/'
save_root = '/home/davidk/Downloads/celeba_dataset/Img/resized_celeba/'
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
    img.save(save_root + 'celebA/' + img_list[i])
    # # img = plt.imread(root + img_list[0])
    # img = imresize(img, (resize_size, resize_size))
    # plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

    if (i % 1000) == 0:
        print('%d images complete' % i)