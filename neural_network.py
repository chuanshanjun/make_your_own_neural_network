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
        hidden_inputs = np.dot(self.wih, inputs_list)
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


input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.5

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

data_file = open('./data/mnist_train_100.csv', 'r')

data_list = data_file.readlines()

data_file.close()

print(len(data_list))

print(data_list[0])

all_values = data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')

all_values2 = data_list[1].split(',')
image_array2 = np.asfarray(all_values2[1:]).reshape((28,28))
plt.imshow(image_array2, cmap='Greys', interpolation='None')


scaled_input = (np.asfarray(all_values[1:])/255.0*0.99) + 0.01
print(scaled_input)

#output nodes is 10 (example)
onodes = 10
targets_demo = np.zeros(onodes) + 0.01
targets_demo[int(all_values[0])] = 0.99
print(targets_demo)
