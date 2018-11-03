import lasagne
from lasagne import layers
import theano.tensor as T
from utils import get_greyscale

class Unpool2DLayer(layers.Layer):

    def __init__(self, incoming, kernel_size, 
        nonlinearity = lasagne.nonlinearities.linear, **kwargs):
      
  super(Unpool2DLayer, self).__init__(incoming, **kwargs)
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity


    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[2] = input_shape[2] * self.kernel_size
        output_shape[3] = input_shape[3] * self.kernel_size
        return tuple(output_shape)


    def get_output_for(self, input, **kwargs):
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return self.nonlinearity(input.repeat(self.kernel_size, axis = 2).repeat(self.kernel_size, axis = 3))
        

class GreyscaleLayer(layers.Layer):
    def __init__(self, incoming, random_greyscale = False, random_seed = 123, **kwargs):
        super(GreyscaleLayer, self).__init__(incoming, **kwargs)
        
        self.rng = T.shared_randomstreams.RandomStreams(random_seed)
        self.random_greyscale = random_greyscale


    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = 1
        return tuple(output_shape)


    def get_output_for(self, input, deterministic_greyscale = False, **kwargs):
        if (not deterministic_greyscale):
            return get_greyscale(input, self.random_greyscale, self.rng)
        return get_greyscale(input, False, self.rng)
