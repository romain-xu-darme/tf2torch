import torch
import torch.nn as nn
import json
from typing import Dict,List

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


def dict2torchlayer (conf: Dict) -> nn.Module:
    """ Given a configuration dictionnary, returns corresponding Pytorch layer
    Args:
        conf (Dict): Layer configuration
    Returns:
        Pytorch nn.Module
    """
    # Recover layer type
    tname = conf['type']
    params = conf['params']

    #################################################
    if tname == "ZeroPad2d":
        left,right,top,bot = params['left'],params['right'],params['top'],params['bot']
        return torch.nn.ZeroPad2d(padding=(left,right,top,bot))

    #################################################
    if tname == "Conv2d":
        pt_layer = torch.nn.Conv2d(
            in_channels = params['in_channels'],
            out_channels = params['out_channels'],
            kernel_size = params['kernel_size'],
            stride = params['stride'],
            padding = params['padding'],
            bias = params['use_bias'],
        )
        # Copy weights
        pt_layer.weight = torch.nn.Parameter(torch.Tensor(params['weights']))
        if params['use_bias']:
            pt_layer.bias.data = torch.Tensor(params['bias'])
        # Handle activation function
        aname = params['activation']
        if aname == 'linear':
            return pt_layer
        if aname == 'relu':
            return torch.nn.Sequential(pt_layer,torch.nn.ReLU())
        if aname == 'sigmoid':
            return torch.nn.Sequential(pt_layer,torch.nn.Sigmoid())
        assert False, f'Activation {aname} not implemented (yet?)'

    #################################################
    if tname == "BatchNorm2d":
        pt_layer = torch.nn.BatchNorm2d(
            num_features = params['num_features'],
            eps = params['eps'],
            momentum = params['momentum'],
        )
        # Copy weights
        pt_layer.gamma = torch.nn.Parameter(torch.Tensor(params['gamma']))
        pt_layer.weight = torch.nn.Parameter(torch.Tensor(params['gamma']))
        pt_layer.beta = torch.nn.Parameter(torch.Tensor(params['beta']))
        pt_layer.bias = torch.nn.Parameter(torch.Tensor(params['beta']))
        pt_layer._buffers['running_mean'] = torch.Tensor(params['mean'])
        pt_layer._buffers['running_var'] = torch.Tensor(params['var'])
        return pt_layer

    #################################################
    if tname == "Activation":
        aname = params['name']
        if aname == "relu":
            return torch.nn.ReLU()
        if aname == "sigmoid":
            return torch.nn.Sigmoid()
        assert False, f'Activation {aname} not implemented (yet?)'

    #################################################
    if tname == "MaxPool2d":
        return torch.nn.MaxPool2d(
            kernel_size = params['kernel_size'],
            stride = params['stride'],
            padding = params['padding']
        )

    #################################################
    if tname == "Add":
        return Add()

    #################################################
    if tname == "Softmax2d":
        return ChannelWiseSoftmax()

    assert False, f'Layer {tname} not implemented (yet?)'


class ModelFromJson (nn.Module):
    def __init__ (self,
        fpath: str,
    ) -> None:
        """ Init model from JSON configuration
        Args:
            fpath (str): Path to JSON file
        """
        super(ModelFromJson,self).__init__()

        # Read configuration file
        with open(fpath,'r') as fin:
            config = json.load(fin)

        # Execution order
        self.exec_order = config['exec']
        self.exec_conf = {}

        # Layer params
        layers_conf = config['layers']
        # Set each layer as attribute
        for name in layers_conf:
            # Build layer from configuration
            setattr(self,name,dict2torchlayer(layers_conf[name]['params']))
            # Store execution confifg
            self.exec_conf[name] = {
                'src': layers_conf[name]['exec']['src'],
                'save': layers_conf[name]['exec']['save'],
            }

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        """ Given the model execution order, process a tensor
        Args:
            x (torch.Tensor): Input tensor
        """
        stack = [x]
        X = stack[0]
        for name in self.exec_order:
            index = self.exec_conf[name]['src']
            if len(index) != 1 or index[0] != -1:
                # Fetch inbound tensors (otherwise, use previous one)
                X = [stack[i] if i != -1 else X for i in index]
                if len(X) == 1: X = X[0]
            X = getattr(self,name)(X)
            if self.exec_conf[name]['save']:
                stack.append(X)
        return X

