import numpy as np

class fcLayer():
    def __init__(self, inSize, outSize, weight_in = None):
        self.inweights = weight_in
        self.insize = inSize
        self.outsize = outSize
        [self.weights, self.bias] = self.init_weight()
        

    def init_weight(self):
        if self.inweights:
            return self.inweights
        else:
            return [np.random.rand(self.insize, self.outsize) - 0.5, np.random.rand(1, self.outsize) - 0.5]


    def forward_propagation(self, inData):
        print("input size = {}, output size = {} \n".format(self.insize, self.outsize))
        self.input = inData
        print("inputs 1 to 10 are: {} \n".format(self.input))
        self.output = np.dot(self.input, self.weights) + self.bias
        print("layer output 1 to 10 are: {} \n".format(self.output))
        return self.output

    def backward_propagation(self, output_err, learning_rate):
        input_err = np.dot(output_err, self.weights.T)
        newinput = np.reshape(self.input,(1,self.insize))
        weights_err = np.matmul(newinput.T, output_err)
        
        self.weights -= learning_rate * weights_err
        self.bias -= learning_rate * output_err
        return input_err

class activationLayer():
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    
    def forward_propagation(self, inData):
        self.input = inData
        # print("inputs 1 to 10 are: {} \n".format(self.input))
        self.output = self.activation(self.input)
        # print("outputs 1 to 10 are: {} \n".format(self.output))
        return self.output

    
    def backward_propagation(self, output_err, learning_rate):
        return self.activation_prime(self.input) * output_err

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def mse(y_true, y_pred):
    #print("y_true= {}, y_pred = {}, diff = {}".format(y_true, y_pred, np.mean(np.power(y_true-y_pred, 2))))
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


class network():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    
    def add(self, layer):
        self.layers.append(layer)
    

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    

    def predict(self, input_data):
        
        samples = len(input_data)
        result = []

        
        for i in range(samples):
            
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    
    def fit(self, x_train_raw, y_train_raw, epochs = 1, learning_rate = 0.9): 
        
        samples = len(x_train_raw)
        lim = int(0.1*samples)
        x_valid = x_train_raw[:lim]
        y_valid = y_train_raw[:lim]
        x_train = x_train_raw[lim:]
        y_train = y_train_raw[lim:]
        train_len = len(x_train)
        valid_len = len(x_valid)
        # x_train = x_train_raw
        # y_train = y_train_raw

        for i in range(epochs):
            for j in range(train_len):
                # print("forward propagation \n")
                output = x_train[j]
                for layer in self.layers:
                    # print("currently in layer: {} \n".format(layer))
                    output = layer.forward_propagation(output)
                
                # print("true label is: {} \n".format(y_train[j]))
                # print("backward propagation \n")
                error = self.loss_prime(y_train[j], output)
                
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            
            for k in range(valid_len):
                err = 0
                valid_output=x_valid[k]
                for layer in self.layers:
                    valid_output = layer.forward_propagation(output)
                
                if(y_valid[k]!=valid_output):
                    err += 1
            
            err /= valid_len
            err *= 100
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
            
                
