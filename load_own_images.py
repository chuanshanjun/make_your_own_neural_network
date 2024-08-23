import imageio.v3
import glob

import numpy as np
import matplotlib.pyplot as plt


# our own image test data set
our_own_dataset = []

for img_file_name in glob.glob('./data/my_own_data/?_0?.png'):
    print("loading ... ", img_file_name)
    # use the filename to set the correct label
    label = int(img_file_name[-8])
    # load image data from png files into an array
    img_array = imageio.v3.imread(img_file_name, mode='F')
    # reshape from 28x28 to list of 784 values, invert values
    img_data = 255 - img_array.reshape(784)
    img_data = (img_data / 255 * 0.99) + 0.01
    print('min val of img_data: ', np.min(img_data))
    print('max val of img_data: ', np.max(img_data))
    # append label and image data to test data set
    record = [label, img_data]
    print('record: ', record)
    our_own_dataset.append(record)
    pass

# import scipy.misc
#
#
# img_array = scipy.misc.imread('mage_file_name', flatten = True)
#
# # 常规而言，0指的是黑色，255指的是白色，但是，MNIST数据集使用相反的方式表示，因此不得不将值逆转过来以匹配MNIST数据。
# img_data = 255.0 - img_array.reshape(255)
# img_data = (img_data / 255.0 * 0.99) + 0.01