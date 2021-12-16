import tensorflow as tf
from typing import Dict
import numpy as np
import json

def tflayer2dict (src: tf.keras.layers) -> Dict:
    """ Returns dictionnary containing configuration of a Tensorflow layer
    Args:
        src (tf.keras.layers): Source layer
    Returns:
        Dictionnary
    """
    # Recover layer type
    tname = src.__class__.__name__.split('.')[-1]

    #################################################
    if tname == "ZeroPadding2D":
        (top, bot), (left,right) = src.padding
        return {'type' : "ZeroPad2d",
                'params' : {
                    'left': left,
                    'right': right,
                    'top': top,
                    'bot': bot
                    }
                }

    #################################################
    if tname == "Conv2D":
        params = {
            'in_channels' : src.input_shape[-1],
            'out_channels' : src.output_shape[-1],
            'kernel_size' : src.kernel_size,
            'stride' : src.strides,
            'padding' : src.padding,
            'use_bias' : src.bias is not None,
            'activation': src.activation.__name__,
            'weights': np.moveaxis(np.moveaxis(src.weights[0],-1,0),-1,1),
        }
        if src.bias is not None:
            params['bias'] = src.weights[1].numpy()
        return {'type': "Conv2d",'params': params}

    #################################################
    if tname == "BatchNormalization":
        params = {
            'num_features': src.gamma.shape[0],
            'eps' : src.epsilon,
            'momentum' : src.momentum,
            'gamma' : src.gamma.numpy(),
            'beta' : src.beta.numpy(),
            'mean' : src.moving_mean.numpy(),
            'var' : src.moving_variance.numpy()
        }
        return {'type': "BatchNorm2d", 'params': params}

    #################################################
    if tname == "Activation":
        return {'type': 'Activation',
                'params': {'name': src.activation.__name__}}

    #################################################
    if tname == "MaxPooling2D":
        # Aaaah, we have to recompute the padding size manually
        assert src.padding == "valid", "'Same' padding not supported (yet?)"
        return {'type': 'MaxPool2d',
                'params': {
                    'padding': 0,
                    'kernel_size': src.pool_size,
                    'stride': src.strides
                    }
                }

    #################################################
    if tname == "Add":
        return {'type': 'Add', 'params' : None}

    #################################################
    if tname == "Softmax":
        # Assume the following TF tensor format: N x H x W x C
        assert src.axis == [1,2], f'Softmax axis support not implemented (yet?): {src.axis}'
        return {'type': 'Softmax2d', 'params': None}

    assert False, f'Layer {tname} not implemented (yet?)'

class NumpyEncoder(json.JSONEncoder):
    """ JSON encoder converting numpy arrays to list for serialization """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def tfmodel2json (source: tf.keras.Model, dest: str) -> None:
    """ Convert a Tensorflow model to a JSON file
    Args:
        source (tf.keras.Model): Source Tensorflow model
        dest (str): Path to JSON file
    """
    layers_conf = {}

    ##############################
    # Build graph
    ##############################
    inputs = []
    graph = {}

    # First pass: Convert and find all inbound nodes for all layers
    for layer in source.layers:
        name = layer.name
        graph[name] = {'inbounds': [], 'outbounds': []}
        metadata = layer._serialized_attributes['metadata']
        if 'inbound_nodes' in metadata.keys():
            # Extract layer information
            layers_conf[name] = {
                    'params': tflayer2dict(layer)
            }
            # Find inbound nodes
            inbounds = metadata['inbound_nodes'][0]
            for ip in range(len(inbounds)):
                inbound = inbounds[ip][0]
                graph[name]['inbounds'].append(inbound)
        else:
            inputs.append(name)

    # Second pass: For each node, find outbound nodes
    for name in graph:
        for inbound in graph[name]['inbounds']:
            graph[inbound]['outbounds'].append(name)

    # Third pass: Find execution order recursively
    exec_order = []
    def add_children(name: str):
        if name in exec_order: return
        for inbound in graph[name]['inbounds']:
            # Missing inbound node, can't execute this node yet
            if inbound not in exec_order: return
        # Inbound nodes already in exec_order list
        exec_order.append(name)
        for outbound in graph[name]['outbounds']:
            add_children(outbound)
        return
    for name in inputs:
        add_children(name)

    # Fourth pass: Compute index of inbound tensors
    stack = inputs.copy()
    for idx, name in enumerate(exec_order):
        exec_conf = {'src': [], 'save': False}
        if name in inputs: continue
        for inbound in graph[name]['inbounds']:
            # Simple case: use previous tensor
            if exec_order[idx-1] == inbound:
                exec_conf['src'].append(-1)
            # Or find index in stack of tensors
            else:
                exec_conf['src'].append(stack.index(inbound))
        # If tensor value is not used straight away, save it
        noutbounds = len(graph[name]['outbounds'])
        if (noutbounds > 1) or \
                (noutbounds == 1 and exec_order[idx+1] != graph[name]['outbounds'][0]) :
            exec_conf['save'] = True
            stack.append(name)
        layers_conf[name]['exec'] = exec_conf

    exec_order = exec_order[len(inputs):] # Skip inputs

    # Save to JSON
    with open('tmp.json','w') as fout:
        json.dump({'layers':layers_conf, 'exec': exec_order},fout,cls=NumpyEncoder)
