from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        
        W1 = np.random.normal(0, weight_scale, input_dim * hidden_dim)
        self.params['W1'] = W1.reshape(input_dim, hidden_dim)
        
        W2 = np.random.normal(0, weight_scale, hidden_dim * num_classes)
        self.params['W2'] = W2.reshape(hidden_dim, num_classes)
        
        self.params['b1'] = np.zeros(hidden_dim)
        
        self.params['b2'] = np.zeros(num_classes)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        layer_1_out, layer_1_cache = affine_relu_forward(X, W1, b1)
        layer_2_out, layer_2_cache = affine_forward(layer_1_out, W2, b2)
        
        scores = layer_2_out
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, d_out = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        dx2, dw2, db2 = affine_backward(d_out, layer_2_cache)
        dx1, dw1, db1 = affine_relu_backward(dx2, layer_1_cache)
        
        dw2 += W2 * self.reg
        dw1 += W1 * self.reg
        
        grads['W2'] = dw2 
        grads['W1'] = dw1
        grads['b1'] = db1
        grads['b2'] = db2
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        
        #do the first layer first because we need to use input_dim
        
        W1 = np.random.normal(0, weight_scale, input_dim * hidden_dims[0])
        #hidden_dim[0] gets the dimensions of the first hidden layer
        self.params['W1'] = W1.reshape(input_dim, hidden_dims[0])
        self.params['b1'] = np.zeros(hidden_dims[0])
        
        for i in range(1, len(hidden_dims)):
            W = np.random.normal(0, weight_scale, hidden_dims[i - 1] * hidden_dims[i])
            self.params['W' + str(i + 1)] = W.reshape(hidden_dims[i - 1], hidden_dims[i])
            self.params['b' + str(i + 1)] = np.zeros(hidden_dims[i])
            
        W_last = np.random.normal(0, weight_scale, hidden_dims[len(hidden_dims) - 1] * num_classes)
        self.params['W' + str(len(hidden_dims) + 1)] = W_last.reshape(hidden_dims[len(hidden_dims) - 1], num_classes)
        self.params['b' + str(len(hidden_dims) + 1)] = np.zeros(num_classes)
        
        
        #initialize batchnorm 
        if self.normalization:
            for i in range(1, self.num_layers):
                self.params['gamma' + str(i)] = np.ones((hidden_dims[i - 1]));
                self.params['beta' + str(i)] = np.ones((hidden_dims[i - 1]));
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        cache_list = []
        data = X.copy() #so we don't lose X
        
        if not self.normalization:
            for i in range(1, self.num_layers):
                out, cache = affine_relu_forward(data, self.params['W' + str(i)], self.params['b' + str(i)])
                #run the forwards step
                cache_list.append(cache) #save cache
                data = out #set the next input

            out, cache = affine_forward(data, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
            cache_list.append(cache)
            scores = out
        
        if self.normalization:
            for i in range(1, self.num_layers):            
                out, cache = affine_bn_relu_forward(data,
                                                    self.params['W' + str(i)],
                                                    self.params['b' + str(i)],
                                                    self.params['beta' + str(i)],
                                                    self.params['gamma' + str(i)],
                                                    self.bn_params[i - 1])
                cache_list.append(cache)
                data = out
            
            out, last_cache = affine_forward(data, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
            cache_list.append(last_cache)
            scores = out
                                     
        #TODO: test this        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        
        dw_s = []
        db_s = []
        dgamma_s = []
        dbeta_s = []
        loss, d_out = softmax_loss(scores, y)
        
        #TODO: implement loss and gradients for batchnorm here
        
        if not self.normalization:
            #last layer first
            cache = cache_list.pop()
            dx, dw, db = affine_backward(d_out, cache)
            dw_s.append(dw)
            db_s.append(db)
            d_out = dx

            for i in range(len(cache_list)):
                cache = cache_list.pop()
                dx, dw, db = affine_relu_backward(d_out, cache)
                dw_s.insert(0, dw) #insert at begining
                db_s.insert(0, db)
                d_out = dx

            add_loss = 0
            for i in range(len(dw_s)):
                W = self.params['W' + str(i + 1)]
                add_loss += np.sum(W * W)
                dw_s[i] += self.reg * W

            loss += 0.5 * self.reg * add_loss

            for i in range(len(dw_s)):
                grads['W' + str(i + 1)] = dw_s[i]
                grads['b' + str(i + 1)] = db_s[i]
        
        if self.normalization: #use batchnorm
            #last layer first
            cache = cache_list.pop()
            dx, dw, db = affine_backward(d_out, cache)
            dw_s.append(dw)
            db_s.append(db)
            d_out = dx
            
            for i in range(len(cache_list)):
                cache = cache_list.pop()
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(d_out, cache)
                dw_s.insert(0, dw)
                db_s.insert(0, db)
                dgamma_s.insert(0, dgamma)
                dbeta_s.insert(0, dbeta)
                d_out = dx
                
            added_loss = 0
            for i in range(len(dw_s)):
                W = self.params['W' + str(i + 1)]
                dw_s[i] += self.reg * W
                added_loss += np.sum(W * W)
              
            for i in range(len(dgamma_s)):
                gamma = self.params['gamma' + str(i + 1)]
                dgamma_s[i] += self.reg * gamma
                beta = self.params['beta' + str(i + 1)]
                dbeta_s[i] += self.reg * beta
                added_loss += np.sum(gamma * gamma) + np.sum(beta * beta)
                                
            loss += 0.5 * self.reg * added_loss
                
            for i in range(len(dw_s)):
                grads['W' + str(i + 1)] = dw_s[i]
                grads['b' + str(i + 1)] = db_s[i]
                    
            for i in range(len(dgamma_s)):
                grads['beta' + str(i + 1)] = dbeta_s[i]
                grads['gamma' + str(i + 1)] = dgamma_s[i]
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    
def affine_bn_relu_forward(x, w, b, beta, gamma, params):
    affine_out, affine_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, params)
    relu_out, relu_cache = relu_forward(bn_out)
    return_cache = (affine_cache, bn_cache, relu_cache)
    return relu_out, return_cache
    
def affine_bn_relu_backward(d_out, cache):
    affine_cache, bn_cache, relu_cache = cache
    d_relu = relu_backward(d_out, relu_cache)
    d_bn, dgamma, dbeta = batchnorm_backward_alt(d_relu, bn_cache)
    dx, dw, db= affine_backward(d_bn, affine_cache)
    return dx, dw, db, dgamma, dbeta

def affine_layernnorm_relu_forward(x, w, b, beta, gamma, params):
    affine_out, affine_cache = affine_forward(x, w, b)
    bn_out, bn_cache = layernorm_forward(affine_out, gamma, beta, params)
    relu_out, relu_cache = relu_forward(bn_out)
    return_cache = (affine_cache, bn_cache, relu_cache)
    return relu_out, return_cache
    
def affine_layernorm_relu_backward(d_out, cache):
    affine_cache, bn_cache, relu_cache = cache
    d_relu = relu_backward(d_out, relu_cache)
    d_bn, dgamma, dbeta = layernorm_backward(d_relu, bn_cache)
    dx, dw, db= affine_backward(d_bn, affine_cache)
    return dx, dw, db, dgamma, dbeta