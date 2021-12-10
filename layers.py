import torch
import torch.nn as nn
import tensorflow as tf
from typing import List
import numpy as np

class Add (nn.Module):
    """ Add two tensors """
    def __init__(self) -> None :
        """ Init layer """
        super(Add,self).__init__()

    def forward(self, X: List[torch.Tensor]) -> torch.Tensor:
        """ Add first two tensors from the list
        Args:
            X (list): List of at least two tensors
        Returns:
            Sum of first two tensors
        """
        assert len(X) >= 2
        return X[0]+X[1]

def convert_layer (src: tf.keras.layers) -> nn.Module:
    """ Given a Tensorflow layer, returns corresponding Pytorch layer
    Args:
        src (tf.keras.layers): Source layer
    Returns:
        Pytorch nn.Module
    """
    # Recover layer type
    tname = src.__class__.__name__.split('.')[-1]

    #################################################
    if tname == "ZeroPadding2D":
        (top, bot), (left,right) = src.padding
        return torch.nn.ZeroPad2d(padding=(left,right,top,bot))

    #################################################
    if tname == "Conv2D":
        pt_layer = torch.nn.Conv2d(
            in_channels = src.input_shape[-1],
            out_channels = src.output_shape[-1],
            kernel_size = src.kernel_size,
            stride = src.strides,
            padding = src.padding,
            bias = src.bias is not None
        )
        # Copy weights
        weights = np.moveaxis(np.moveaxis(src.weights[0],-1,0),-1,1)
        pt_layer.weight = torch.nn.Parameter(torch.Tensor(weights))
        if src.bias is not None:
            pt_layer.bias.data = torch.Tensor(
                src.weights[1].numpy())
        # Handle activation function
        aname = src.activation.__name__
        if aname == 'linear':
            return pt_layer
        if aname == 'relu':
            return torch.nn.Sequential(pt_layer,torch.nn.ReLU())
        assert False, f'Activation {aname} not implemented (yet?)'
        return pt_layer

    #################################################
    if tname == "BatchNormalization":
        pt_layer = torch.nn.BatchNorm2d(
            num_features = src.gamma.shape[0],
            eps = src.epsilon,
            momentum = src.momentum,
        )
        # Copy weights
        pt_layer.gamma = torch.nn.Parameter(torch.Tensor(src.gamma.numpy()))
        pt_layer.weight = torch.nn.Parameter(torch.Tensor(src.gamma.numpy()))
        pt_layer.beta = torch.nn.Parameter(torch.Tensor(src.beta.numpy()))
        pt_layer.bias = torch.nn.Parameter(torch.Tensor(src.beta.numpy()))
        pt_layer._buffers['running_mean'] = torch.Tensor(src.moving_mean.numpy())
        pt_layer._buffers['running_var'] = torch.Tensor(src.moving_variance.numpy())
        return pt_layer

    #################################################
    if tname == "Activation":
        aname = src.activation.__name__
        if aname == "relu":
            return torch.nn.ReLU()
        assert False, f'Activation {aname} not implemented (yet?)'

    #################################################
    if tname == "MaxPooling2D":
        if src.padding == "valid":
            padding = 0
        else:
            # Aaaah, we have to recompute the padding size manually
            assert False, "'Same' padding not supported (yet?)"

        return torch.nn.MaxPool2d(
            kernel_size = src.pool_size,
            stride = src.strides,
            padding = padding,
        )

    #################################################
    if tname == "Add":
        return Add()

    assert False, f'Layer {tname} not implemented (yet?)'


