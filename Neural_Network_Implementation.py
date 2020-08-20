# Neural Network Implementation
"""

import numpy as np
import h5py
import random
import matplotlib.pyplot as plt


"""### Activation Functions"""

class Activation(object):
    def tanh(self, x):
        return np.tanh(x) 
  
    def tanh_deriv(self, a): 
        return 1.0 - a**2

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_deriv(self, a):   
        return a*(1 - a)
    
    def relu(self, x):       
        return np.maximum(0,x)
    
    def relu_deriv(self, a):    
        a[a <= 0] = 0
        a[a > 0] = 1
        return a 
    
    def softmax(self, x):     
        exps = np.exp(x - np.max(x)) #for stability 
        exps = exps / np.sum(exps)
        return exps 
    
    def softmax_deriv(self, a):
        a = a.reshape(-1,1)
        return np.diagflat(a) - np.dot(a, a.T)  
    
    def __init__(self, activation = 'tanh'):
        if activation == 'tanh':
            self.function = self.tanh
            self.deriv = self.tanh_deriv

        if activation == 'sigmoid':
            self.function = self.sigmoid
            self.deriv = self.sigmoid_deriv 

        if activation == 'relu':
            self.function = self.relu
            self.deriv = self.relu_deriv 

        if activation == 'softmax':
            self.function = self.softmax
            self.deriv = self.softmax_deriv

"""## Batch Normalisation Class"""

class BatchNormLayer(object): 
    
    def __init__(self, momentum = 0.90): 
        
        #momentum term used to calculate running mean & variance. 
        self.momentum = momentum
        
        self.grad_gamma = 0
        self.grad_beta = 0
        
        #initialise gamma and beta 
        self.gamma = np.random.uniform(low = 0, high = 1)
        self.beta = np.random.uniform(low = 0, high = 1)

        #initialise retained gradients from previous iterations. Used for momentum. 
        self.v_gamma = np.zeros_like(self.gamma)
        self.v_beta = np.zeros_like(self.beta)
        
        self.x_norm = 0 #normalised output after applying batch norm.
        self.std = 0 #standard deviation of inputs fed into batch norm layer. 

        #running mean and variance to be used when batch norm is applied in test mode.
        self.running_mean = 0
        self.running_var = 0 
        
        #this is the dropout mask. For the batch norm layer, we don't apply dropout
        #so set this to None. 
        self.mask = None 
        
    def batch_norm_forward(self, input, mode): 
        """
        Forward pass for the batch normalisation layer. 

        Parameters:
            input (numpy array): activation output from previous layer. 
            mode (string): whether we are using the neural network for train or testing. 

        Returns: 
            self.output (numpy array): normalised and scaled output from the batch norm layer.
        """
        momentum = self.momentum
        
        if mode == 'train': 
            
            sample_mean = np.mean(input)
            sample_var = np.var(input)
            
            #update running mean and variance.
            self.running_mean = momentum * self.running_mean + (1-momentum) * sample_mean
            self.running_var = momentum * self.running_var + (1-momentum) * sample_var
            
            #update normalised output and standard deviation of input.
            input_centered = (input - sample_mean) 
            self.std = np.sqrt(sample_var + 1e-5) 
            self.x_norm = input_centered/self.std 
            
            #output of batch normalisation layer. 
            self.output = self.gamma * self.x_norm + self.beta 
        
        elif mode == 'test': 
            #in test mode, use the stored running mean and variance to calculate normalised output
            x_norm = (input - self.running_mean) / np.sqrt(self.running_var + 1e-5)
            self.output = self.gamma * x_norm + self.beta
        
        return self.output 
    
    def batch_norm_backward(self, dy, minibatch = False):
        """
        Backward pass for the batch normalisation layer calculated based 
        off analytical derivative. 

        Parameters:
            dy(numpy array): incoming gradient from the layer above 
            minibatch (boolean): whether we are using mini-batch training or not. 

        Returns: 
            dx (numpy array): gradient to be used as input to the backpass of the layer below.
        """
        N = dy.shape[0]
        
        #update gradients of gamma and beta. 
        dgamma = np.sum(dy * self.x_norm)
        dbeta = np.sum(dy)
        
        dx_norm = dy * self.gamma
        
        first_term = N * dx_norm
        second_term = np.sum(dx_norm)
        third_term = self.x_norm * np.sum(dx_norm * self.x_norm)
        
        dx = 1/N/self.std * (first_term - second_term - third_term)
        
        #if using minibatch training, keep adding on the gradients for gamma and beta.
        if minibatch == False:
            self.grad_gamma = dgamma
            self.grad_beta = dbeta 
        else: 
            self.grad_gamma += dgamma 
            self.grad_beta += dbeta 
        
        return dx

"""## Hidden Layer Class"""

class HiddenLayer(object):
    def __init__(self,n_in, n_out,
               activation_last_layer='tanh',activation='tanh', initialisation = 'xavier', 
               W = None, b = None, mask = None):
        
        self.input = None 
        self.activation = Activation(activation).function
    

        # activation deriv of last layer
        self.activation_deriv=None
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).deriv
    
        #initialise weights
        if initialisation == 'xavier': #recommended to use with tanh
            self.W = np.random.uniform(
              low = -np.sqrt(6. / (n_in + n_out)),
              high = np.sqrt(6. / (n_in + n_out)),
              size = (n_in, n_out))
    
        if initialisation == 'kaiming': #recommended to use with relu
            self.W = np.random.uniform(
              low = -np.sqrt(6./n_in), #let fan_mode be input size 
              high = np.sqrt(6./n_in),
              size = (n_in, n_out))
      
        if activation == 'sigmoid': #this is done to avoid vanishing gradients. 
            self.W *= 4

        self.b = np.zeros(n_out,) #initialise bias 

        #initialise gradients of weights and bias 
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)
        
        #initialise retained gradients from previous iterations. Used for momentum in SGD. 
        self.v_W = np.zeros(self.W.shape)
        self.v_b = np.zeros(self.b.shape)
        
        #dropout mask for this hidden layer. 
        self.mask = None 
                
    def dropout(self, output, dropout_prob):
        '''
        Applies inverted dropout to the activations by multiplying the activation 
        output of the layer by a mask. Outputs are zeroed out at the point where the 
        mask contains zero. Retained outputs are scaled by the appropriate factor. 

        Parameters: 
            output (numpy array): output from hidden layer without dropout applied.
            dropout_prob (float): dropout probability

        Returns: 
            output (numpy array): output from hidden layer after applying dropout.
        '''
        keep_prob = 1 - dropout_prob
        #this is a numpy array containing zeros at places where outputs should be zeroed out.
        mask = np.random.binomial(1, keep_prob, size = output.shape) / keep_prob      
        self.mask = mask 
        output *= mask
        
        return output
    
    def forward(self, input, dropout_prob, mode = 'train'):
        '''
        Calculates the output of the forward pass for this hidden layer.

        Parameters:
            input (numpy array): output from previous layer. 
            dropout_prob (float): dropout probability.
            mode (string): whether we are training or testing mode. 

        Returns:
            self.output (numpy array): output of this layer after applying dropout.
        '''
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output))
        self.input = input
        
        if dropout_prob is not None and mode == 'train': #use dropout only in train mode.
            self.output = self.dropout(self.output, dropout_prob)
        
        return self.output

    def backward(self, delta, mask, minibatch = False, lambda_reg = None, output_layer=False):
        '''
        Calculates the delta of the backward pass for this hidden layer.

        Parameters:
            delta (numpy array): delta from layer above.
            mask (numpy array): dropout mask used for the incoming inputs to this layer. Set
                                to none if no dropout is used. 
            minibatch (boolean): whether minibatch training is used or not.
            lambda_reg: the lambda used for weight decay. Set to none if no weight decay is used.
            output_layer (boolean): whether the current layer is an output layer or not.

        Returns:
            delta (numpy array): delta to be used as input to the layer below in the back pass.
        '''
        
        #if minibatch is used, keep adding on the gradients of weights and bias
        if minibatch == True:
            self.grad_W += np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
            self.grad_b += delta
        else:
            self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
            self.grad_b = delta

        #if weight decay used, adjust gradients of weights.
        if lambda_reg is not None:
            self.grad_W += lambda_reg * self.W 
        
        if self.activation_deriv: 
            #if dropout is used, the gradients of the zeroed out inputs should not be flowed back. 
            if mask is not None: 
                delta = delta.dot(self.W.T) * self.activation_deriv(self.input) * mask 
            else:
                delta = delta.dot(self.W.T) * self.activation_deriv(self.input) 
        return delta

"""## The MLP"""

class NeuralNetwork:
    
    def __init__(self, layers, activation=[None,'tanh','tanh'], batch_norm = [False, False, False],
                 initialisation = 'xavier', dropout_prob = None, lambda_reg = None, sgd_momentum = None):

        ### initialize layers
        self.layers = []
        self.params = []
        
        #weight decay and momentum terms. 
        self.lambda_reg = lambda_reg
        self.sgd_momentum = sgd_momentum 

        #stores which layers use batch normalised outputs. 
        self.batch_norm = batch_norm

        #activation and weight initialisation
        self.activation = activation
        self.initialisation = initialisation

        #the percentage of units to randomly drop out at each layer. 
        self.dropout_prob = dropout_prob
    
        for i in range(len(layers)-1): 
            hidden = HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1], initialisation)
            if batch_norm[i] == True: #create an extra layer of batch norm class if batch norm is used.
                bn_hidden = BatchNormLayer() 
                self.layers.append(bn_hidden)
            self.layers.append(hidden)
      
    def forward(self,input, mode = 'train'):
        '''
        Foward passes the input throughout the neural network. 

        Parameters:
            input (numpy array): the input sample data
            mode (string): whether we are in training or testing mode.
        Returns:
            output (numpy array): output of the entire neural network. 
        '''
        for layer in self.layers:
            #use batch_norm_forward if layer is a batch norm layer. 
            if isinstance(layer, BatchNormLayer): 
                output = layer.batch_norm_forward(input, mode)
            #do not apply dropout to last (output) layer
            elif layer == self.layers[-1]: 
                output = layer.forward(input, None, mode)
            else: 
                output = layer.forward(input, self.dropout_prob, mode)
            input = output
        return output
    
    def regularization_term(self):
        '''
        Calculates the l2 regularisation term that is added to the loss
        function when using weight decay.

        Returns: 
            l2_regularisation (float): the regularisation term. 
        '''
        l2_regularisation = 0.0

        for layer in self.layers:
            if isinstance(layer, BatchNormLayer): 
                continue 
            l2_regularisation += np.sum(np.power(layer.W, 2))

        return l2_regularisation
    
    def criterion_cross_entropy(self, y, y_hat):
        '''
        Calculates the cross entropy (CE) loss for the predictions.

        Parameters: 
            y (numpy array): one hot encoded ground truth value vector
            y_hat (numpy array): neural network predictions.

        Returns:
            loss (float): the CE loss between y and y_hat
            delta (numpy array): delta of output layer. 
        '''
        activation_deriv = Activation(self.activation[-1]).deriv
        loss = -np.sum(y*np.log(y_hat + 1e-8)) #cross entropy loss
        if self.lambda_reg is not None: #regularised loss
            penalty = self.lambda_reg/2*self.regularization_term()
            loss += penalty
        delta = y_hat - y  #this is the analytical derivative of softmax + CE 
        return loss, delta
    
    def backward(self,delta, minibatch = False):
        '''
        This function does the backpropagation process. 

        Parameters:
            delta (numpy array): delta of the output layer 
            minibatch (boolean): whether minibatch training is used or not. 
        '''
        #delta of the last layer
        delta = self.layers[-1].backward(delta, self.layers[-2].mask, minibatch, output_layer = True)
        
        for i, layer in reversed(list(enumerate(self.layers[:-1]))):
            if isinstance(layer, BatchNormLayer):
                delta = layer.batch_norm_backward(delta, minibatch)
            elif i == 0: #this is the input layer, don't apply dropout on it 
                delta = layer.backward(delta, None, minibatch)
            else: 
                delta = layer.backward(delta, self.layers[i-1].mask, minibatch, self.lambda_reg)

      
    def update(self,lr, minibatch_size): 
        '''
        Update the parameters of the network. If the minibatch_size is 1, we are using
        stochastic gradient descent, otherwise we are using minibatch training. 

        Parameters:
            lr (float): the learning rate.
            minibatch_size (integer): the size of each batch. 
        '''
        if self.sgd_momentum is not None: #momentum is used 
            for layer in self.layers:  
                if isinstance(layer, BatchNormLayer): 
                    layer.v_gamma = layer.v_gamma * self.sgd_momentum + lr * (1/minibatch_size)*layer.grad_gamma 
                    layer.v_beta = layer.v_beta * self.sgd_momentum + lr * (1/minibatch_size)*layer.grad_beta 
                    
                    layer.gamma = layer.gamma - layer.v_gamma
                    layer.beta = layer.beta - layer.v_beta
                else: 
                    layer.v_W = layer.v_W * self.sgd_momentum + lr * (1/minibatch_size)*layer.grad_W 
                    layer.v_b = layer.v_b * self.sgd_momentum + lr * (1/minibatch_size)*layer.grad_b 
                    
                    ##update weights and bias 
                    layer.W = layer.W - layer.v_W
                    layer.b = layer.b - layer.v_b
        else:
            for layer in self.layers: 
                if isinstance(layer, BatchNormLayer): #it is a batch norm layer 
                    layer.gamma -= lr * (1/minibatch_size) * layer.grad_gamma 
                    layer.beta -= lr * (1/minibatch_size) * layer.grad_beta
                else: 
                    layer.W -= lr * (1/minibatch_size) * layer.grad_W
                    layer.b -= lr * (1/minibatch_size) * layer.grad_b 
      
    def sgd(self, X, y, learning_rate, epochs): 
        '''
        Use stochastic gradient descent to train the model. 

        Parameters:
            X (numpy array): Input data or features
            y (numpy array): Input targets
            learning_rate (float): parameters defining the speed of learning
            epochs (integer): number of times the dataset is presented to the network for learning

        Returns: 
            accuracy_to_return (numpy array): training accuracy for each epoch
            loss_to_return (numpy array): average loss for each epoch
        '''
        X = np.array(X)
        y = np.array(y)

        loss_to_return = np.zeros(epochs)
        accurary_to_return = np.zeros(epochs)
        
        for k in range(epochs):
            print(k+1)

            losses = []
            accuracy = 0
            
            for it in range(X.shape[0]):
                #generate random numbers less than 
                i = np.random.randint(X.shape[0]) 

                y_hat = self.forward(X[i]) #forward pass
                y_true = np.zeros(10) #one-hot encode ground truth
                y_true[y[i]] = 1 
                loss,delta = self.criterion_cross_entropy(y_true,y_hat)
                self.backward(delta) #backpass 
                
                losses.append(loss)
                predicted = np.argmax(y_hat)
                true = y[i]

                if predicted == true:
                    accuracy += 1

                #update parameters. For SGD, set the batch size to 1
                self.update(learning_rate, minibatch_size = 1)

            loss_to_return[k] = np.mean(loss)
            accurary_to_return[k] = accuracy/X.shape[0]

            print('Epoch Accuracy:', accurary_to_return[k])
            print('Epoch Average Loss:', loss_to_return[k])

        return accurary_to_return, loss_to_return

    def mini_batch(self,X,y,minibatch_size,learning_rate, epochs):
        '''
        Use mini-batch gradient descent to train the model. 

        Parameters:
            X (numpy array): Input data or features
            y (numpy array): Input targets
            learning_rate (float): parameters defining the speed of learning
            epochs (integer): number of times the dataset is presented to the network for learning

        Returns: 
            accuracy_to_return (numpy array): training accuracy for each epoch
            loss_to_return (numpy array): average loss for each epoch
        '''
        X = np.array(X)
        y = np.array(y)
        data = list(zip(X,y))

        loss_to_return = np.zeros(epochs)
        accurary_to_return = np.zeros(epochs)
        
        for k in range(epochs):
            print('Epoch:', k+1)
            
            losses = []

            random.shuffle(data) #randomise data 
            X_train, y_train = zip(*data)
            X_train, y_train = np.array(X_train), np.array(y_train)
            
            accuracy = 0
            
            for it in range(0, X_train.shape[0], minibatch_size):

                #create batches according to batch size 
                X_train_mini = X_train[it:it + minibatch_size]
                y_train_mini = y_train[it:it + minibatch_size]

                for i in range(0, minibatch_size):
                    
                    y_hat = self.forward(X_train_mini[i]) #a vector of probabilities
                    y_true = np.zeros(10) #one hot encode ground truth
                    y_true[y_train_mini[i]] = 1 
                    
                    loss, delta = self.criterion_cross_entropy(y_true,y_hat)
                    losses.append(loss)
                    predicted = np.argmax(y_hat)
                    true = y_train_mini[i]

                    if predicted == true: 
                        accuracy += 1
                    
                    if i == 0: #first of the batch; don't start adding gradients together yet 
                        self.backward(delta, minibatch = False)
                    else: 
                        self.backward(delta, minibatch = True)

                # update after each minibatch 
                self.update(learning_rate, minibatch_size)

                if it % 5000 == 0:
                    print('Number of examples processed: ', it + 5000, ' Loss: ', losses[-1])
    
            loss_to_return[k] = np.nanmean(losses)
            accurary_to_return[k] = accuracy/X.shape[0]

            print('Epoch Accuracy:', accurary_to_return[k])
            print('Epoch Average Loss:', loss_to_return[k])
        return accurary_to_return, loss_to_return
    
    def fit(self,X,y,learning_rate=0.1, epochs=100, minibatch_size = None):
        '''
        Fit the model using either stochastic or mini-batch gradient descent.
        '''
        if minibatch_size is None:
            return self.sgd(X, y, learning_rate, epochs)
        else:
            return self.mini_batch(X,y,minibatch_size,learning_rate, epochs)

    def predict(self, x):
        '''
        Using the trained model, predict on unseen data. 
        '''
        x = np.array(x)
        output = []
        for i in np.arange(x.shape[0]):
            y_hat = nn.forward(x[i], mode = 'test')
            output.append(np.argmax(y_hat))
        return output
