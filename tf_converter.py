import torch
import torch.nn as nn
import tensorflow as tf
from typing import List, Dict, Tuple
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

class ChannelWiseSoftmax(nn.Module):
    """ Channel-wise Softmax
    When given a tensor of shape ``N x C x H x W``, apply
    Softmax to each channel
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4, 'ChannelWiseSoftmax requires a 4D tensor as input'
        # Reshape tensor to N x C x (H x W)
        reshaped  = input.view(*input.size()[:2], -1)
        # Apply softmax along 2nd dimension than reshape to original
        return nn.Softmax(2)(reshaped).view_as(input)

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
        if aname == 'sigmoid':
            return torch.nn.Sequential(pt_layer,torch.nn.Sigmoid())
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
        if aname == "sigmoid":
            return torch.nn.Sigmoid()
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

    #################################################
    if tname == "Softmax":
        # Assume the following TF tensor format: N x H x W x C
        naxis = len(src.axis)
        assert src.axis == [1,2], f'Softmax axis support not implemented (yet?): {src.axis}'
        return ChannelWiseSoftmax()

    assert False, f'Layer {tname} not implemented (yet?)'

def convert_model (source: tf.keras.Model) -> Tuple[Dict,List,Dict]:
    """ Convert a Tensorflow model
    Args:
        source (tf.keras.Model): Source Tensorflow model
    Returns:
        Dictionnary containing layer names and associated nn.Module
        List of layer names corresponding to execution order during forward pass
        Dictionnary containing execution configuration for each layer
    """
    ##############################
    # Build graph
    ##############################
    inputs = []
    layers_conf = {}

    # First pass: Convert and find all inbound nodes for all layers
    layers_func = {}
    for layer in source.layers:
        name = layer.name
        layers_conf[name] = {}
        layers_conf[name]['inbounds'] = []
        layers_conf[name]['outbounds'] = []
        metadata = layer._serialized_attributes['metadata']
        if 'inbound_nodes' in metadata.keys():
            # Convert TF layer into Torch
            layers_func[name] = convert_layer(layer)
            # Find inbound nodes
            inbounds = metadata['inbound_nodes'][0]
            for ip in range(len(inbounds)):
                inbound = inbounds[ip][0]
                layers_conf[name]['inbounds'].append(inbound)
        else:
            inputs.append(name)

    # Second pass: For each node, find outbound nodes
    for name in layers_conf:
        for inbound in layers_conf[name]['inbounds']:
            layers_conf[inbound]['outbounds'].append(name)

    # Third pass: Find exec_order exec_order recursively
    exec_order = []
    def add_children(name: str):
        if name in exec_order: return
        for inbound in layers_conf[name]['inbounds']:
            # Missing inbound node, can't execute this node yet
            if inbound not in exec_order: return
        # Inbound nodes already in exec_order list
        exec_order.append(name)
        for outbound in layers_conf[name]['outbounds']:
            add_children(outbound)
        return
    for name in inputs:
        add_children(name)

    # Fourth pass: Compute index of inbound tensors
    exec_conf = {}
    stack = inputs.copy()
    for idx, name in enumerate(exec_order):
        exec_conf[name] = {'src_index': [], 'save': False}
        if name in inputs: continue
        for inbound in layers_conf[name]['inbounds']:
            # Simple case: use previous tensor
            if exec_order[idx-1] == inbound: exec_conf[name]['src_index'].append(-1)
            # Or find index in stack of tensors
            else:
                exec_conf[name]['src_index'].append(stack.index(inbound))
        # If tensor value is not used straight away, save it
        noutbounds = len(layers_conf[name]['outbounds'])
        if (noutbounds > 1) or \
                (noutbounds == 1 and exec_order[idx+1] != layers_conf[name]['outbounds'][0]) :
            exec_conf[name]['save'] = True
            stack.append(name)

    exec_order = exec_order[len(inputs):] # Skip inputs
    return layers_func, exec_order, exec_conf
