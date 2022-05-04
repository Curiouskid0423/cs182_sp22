# from turtle import shape
import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################

        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        # Conv layer * 1
        self.params['W1'] = np.random.normal(
            scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
        self.params['b1'] = np.zeros(shape=(num_filters,))
        
        # Affine layers * 2
        input_h, input_w = input_dim[1], input_dim[2]
        pad = self.conv_param['pad']
        ph, pw = self.pool_param['pool_height'], self.pool_param['pool_width']
        conv_out_height = 1 + (input_h + 2 * pad - filter_size) // self.conv_param['stride']
        conv_out_width = 1 + (input_w + 2 * pad - filter_size) // self.conv_param['stride']
        pool_out_height = 1 + (conv_out_height - ph) // self.pool_param['stride']
        pool_out_width = 1 + (conv_out_width - pw) // self.pool_param['stride']
        flatten_dim = pool_out_width * pool_out_height * num_filters
        
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(flatten_dim, hidden_dim))
        self.params['b2'] = np.zeros(shape=(hidden_dim,))
        self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros(shape=(num_classes,))
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        cache = {}
        feat, cache['conv1'] = conv_forward_fast(X, self.params['W1'], self.params['b1'], self.conv_param)
        feat, cache['relu1'] = relu_forward(feat)
        feat, cache['pool1'] = max_pool_forward_naive(feat, self.pool_param)
        pre_flatten_shape = feat.shape
        feat = feat.reshape(feat.shape[0], -1)

        feat, cache['aff2'] = affine_forward(feat, self.params['W2'], self.params['b2'])
        feat, cache['relu2'] = relu_forward(feat)
        scores, cache['aff3'] = affine_forward(feat, self.params['W3'], self.params['b3'])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        if self.reg > 0:
          l2_norm = 0
          for key, val in self.params.items():
            if "W" in key:
              l2_norm += np.linalg.norm(val) ** 2
          loss += l2_norm * self.reg * 0.5
        dout, grads['W3'], grads['b3'] = affine_backward(dout, cache['aff3'])
        grads['W3'] += self.params['W3'] * self.reg

        dout = relu_backward(dout, cache['relu2'])
        dout, grads['W2'], grads['b2'] = affine_backward(dout, cache['aff2'])
        grads['W2'] += self.params['W2'] * self.reg
        
        dout = np.reshape(dout, newshape=pre_flatten_shape)

        dout = max_pool_backward_naive(dout, cache['pool1'])
        dout = relu_backward(dout, cache['relu1'])
        dout, grads['W1'], grads['b1'] = conv_backward_fast(dout, cache['conv1'])
        grads['W1'] += self.params['W1'] * self.reg

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
