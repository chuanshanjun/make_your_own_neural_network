import numpy as np
import scipy.special

# neural network class definition
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.innodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learnning rate
        self.lr = learningrate

        # 权重最好在-1.0 -> 1.0 之间，这边偷懒 -5 使权重在 -0.5 -> 0.5 之间
        # 权重是不会随着方法的调用而消失的，它与神经网络一起，所以权重一开始就要初始化
        self.wih = (np.random.rand(3,3) - 0.5)
        self.who = (np.random.rand(3,3) - 0.5)

        # activation function

    # train the neural network
    def train(self):
        pass

    # X(hidden) = W(input_hidden)*I
    # query the neural network
    def query(self):
        pass


# number of input, hidden and output nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

# learning rate is 0.5
learning_rate = 0.5

# create instance of neural networkd
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)









