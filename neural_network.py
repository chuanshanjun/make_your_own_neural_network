import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

        self.activation_function = lambda x: expit(x)

        pass

    def train(self, inputs_list, target_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # 从输入层前馈信号到最终输出层
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr*np.dot((output_errors*final_outputs*(1-final_outputs)), np.transpose(hidden_outputs))

        self.wih += self.lr*np.dot(hidden_errors*hidden_outputs*(1-hidden_errors), np.transpose(inputs))

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
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
# training_data_file = open('./data/mnist_train_100.csv', 'r')
training_data_file = open('./data/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# go through all records in the training data set
for record in training_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    #scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create the target output values (all 0.01, except the desired label which is 0.99)
    targets = np.zeros(output_nodes) + 0.01
    # all_values[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
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
    print(correct_label, "correct label")
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    print(label, "network's answer")
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
print("performace = ", scorecard_array.sum() / scorecard_array.size)

# 打印单个测试数据集以及查看网络输出
# all_values = test_data_list[0].split(',')
# print(all_values[0])

# image_array = np.asfarray(all_values[1:]).reshape((28,28))
# plt.imshow(image_array, cmap='Greys', interpolation='None')

# print(n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))

