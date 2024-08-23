import numpy as np
import matplotlib.pyplot as plt

import imageio.v3
import glob



# our own image test data set
our_own_dataset = []

for img_file_name in glob.glob('./data/my_own_data/?_0?.png'):
    # use the filename to set the correct label
    label = int(img_file_name[-8])

    # load image data from png files into an array
    print("loading ... ", img_file_name)
    img_array = imageio.v3.imread(img_file_name, mode='F')

    # use the filename to set the correct label
    label = int(img_file_name[-8])
    # load image data from png files into an array

    # reshape from 28x28 to list of 784 values, invert values
    img_data = 255 - img_array.reshape(784)

    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255 * 0.99) + 0.01
    print('min val of img_data: ', np.min(img_data))
    print('max val of img_data: ', np.max(img_data))

    # append label and image data to test data set
    record = np.append(label,img_data)
    our_own_dataset.append(record)
    pass

# test the neural network with our own images

# record to test
item = 0

plt.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

correct_label = our_own_dataset[item][0]
inputs = our_own_dataset[item][1:]