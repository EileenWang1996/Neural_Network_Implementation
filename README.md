# Neural_Network_Implementation

This allowed me to learn how simply neural networks and backpropagation works. The code also includes implementation of regularisation methods such as dropout, weight decay, and batch normalisation. There is a choice of two optimiser: stochastic gradient descent (SGD) and mini-batch gradient descent. To initialise the Neural Network class, 7 arguments are required to be inputted. The details are below. 

**1st argument:** each element in list represents a layer with a specified number of hidden units. First layer should is the input layer equal to the number of features in the data.

**2nd argument:** the activation function used for each layer. Either 'sigmoid', 'tanh', 'relu' or 'softmax'.

**3rd argument:** whether to use batch normalisation on the outputs for that layer. True if you want to use batch norm, otherwise False.

**4th argument:** which weight initlisation method to use. Either 'kaiming' or 'xavier'. 

**5th argument:** the dropout probability. Set it to None if you don't want to use dropout.

**6th argument:** the lambda regularisation term. Set to None if you don't want to use weight decay.

**7th argument:** the SGD momentum term. Set to None if you don't want to use momentum.
