import numpy as np
import scipy.special
import matplotlib.pyplot as plt

import imageio.v3
import glob



class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        # 两种权重生成方式，对整个模型提升率不大，下面“正态概率分布采样权重” 大约提高了 1% 不到
        # 毕竟最后的权重会被数据正确的训练即可
        # self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 从输入层前馈信号到最终输出层
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr*np.dot((output_errors*final_outputs*(1.0 - final_outputs)), np.transpose(hidden_outputs))

        self.wih += self.lr*np.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)), np.transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
# training_data_file = open('./data/mnist_train_100.csv', 'r')
training_data_file = open('./data/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# epochs is the number of times the training data set is used for train
epochs = 5
for e in range(epochs):
    # go through all records in the training data set
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass


# load the mnist test data CSV file into a list
# test_data_file = open('./data/mnist_test_10.csv', 'r')
test_data_file = open('./data/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # print(correct_label, "correct label")
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # print(label, "network's answer")
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    pass

print(scorecard, "scorecard")

# calculate the performacne score, the fraction of correct answers
scorecard_array = np.asfarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)


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

outputs = n.query(inputs)
print(outputs)

label = np.argmax(outputs)
print('network says: ', label)

if label == correct_label:
    print("match!")
else:
    print("no match")
    pass