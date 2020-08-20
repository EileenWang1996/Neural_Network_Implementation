# Neural_Network_Implementation

This allowed me to learn how simple neural networks and backpropagation works. The code also includes implementation of regularisation methods such as dropout, weight decay, and batch normalisation. There is a choice of two optimisers: stochastic gradient descent (SGD) and mini-batch gradient descent. To initialise the Neural Network class, 7 arguments are required to be inputted. The details are below. 

**1st argument:** each element in list represents a layer with a specified number of hidden units. First layer should is the input layer equal to the number of features in the data.

**2nd argument:** the activation function used for each layer. Either 'sigmoid', 'tanh', 'relu' or 'softmax'.

**3rd argument:** whether to use batch normalisation on the outputs for that layer. True if you want to use batch norm, otherwise False.

**4th argument:** which weight initlisation method to use. Either 'kaiming' or 'xavier'. 

**5th argument:** the dropout probability. Set it to None if you don't want to use dropout.

**6th argument:** the lambda regularisation term. Set to None if you don't want to use weight decay.

**7th argument:** the SGD momentum term. Set to None if you don't want to use momentum.

# Example code that initialises the Neural Network class

```python
### initialise the model 
nn = NeuralNetwork([128, 64, 10], [None, 'relu','softmax'], 
                   [False, True, False], 'kaiming', dropout_prob = 0.50, 
                   lambda_reg = 0.001, sgd_momentum = None)
input_data = data
output_data = label

### create validation and train set
train_data = input_data[0:48000]
train_output = output_data[0:48000]
validation_set = input_data[48000:]
validation_output = output_data[48000:]

### fit the network 
CE_Accuracy,CE_loss = nn.fit(train_data , train_output, learning_rate=0.015, 
                             epochs= 30, minibatch_size = 100)
```
