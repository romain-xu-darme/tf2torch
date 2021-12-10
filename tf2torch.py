import tensorflow as tf
import torch
import torch.nn as nn
from layers import convert_layer

class TFConvertedModel (nn.Module):
    def __init__ (self,
        source: tf.keras.Model,
    ) -> None:
        """ Init Pytorch Model
        Args:
            source (tf.keras.Model): Source Tensorflow model
        """
        super(TFConvertedModel,self).__init__()

        ##############################
        # Build graph
        ##############################
        inputs = []
        layers = {}

        # First pass: Convert and find all inbound nodes for all layers
        for layer in source.layers:
            name = layer.name
            layers[name] = {}
            layers[name]['type'] = type(layer)
            layers[name]['inbounds'] = []
            layers[name]['outbounds'] = []
            metadata = layer._serialized_attributes['metadata']
            if 'inbound_nodes' in metadata.keys():
                # Convert TF layer into Torch
                setattr(self,name,convert_layer(layer))
                # Find inbound nodes
                inbounds = metadata['inbound_nodes'][0]
                for ip in range(len(inbounds)):
                    inbound = inbounds[ip][0]
                    layers[name]['inbounds'].append(inbound)
            else:
                inputs.append(name)

        # Second pass: For each node, find outbound nodes
        for name in layers:
            for inbound in layers[name]['inbounds']:
                layers[inbound]['outbounds'].append(name)

        # Third pass: Find exec_order exec_order recursively
        exec_order = []
        def add_children(name: str):
            if name in exec_order: return
            for inbound in layers[name]['inbounds']:
                # Missing inbound node, can't execute this node yet
                if inbound not in exec_order: return
            # Inbound nodes already in exec_order list
            exec_order.append(name)
            for outbound in layers[name]['outbounds']:
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
            for inbound in layers[name]['inbounds']:
                # Simple case: use previous tensor
                if exec_order[idx-1] == inbound: exec_conf[name]['src_index'].append(-1)
                # Or find index in stack of tensors
                else:
                    exec_conf[name]['src_index'].append(stack.index(inbound))
            # If tensor value is not used straight away, save it
            noutbounds = len(layers[name]['outbounds'])
            if (noutbounds > 1) or \
                    (noutbounds == 1 and exec_order[idx+1] != layers[name]['outbounds'][0]) :
                exec_conf[name]['save'] = True
                stack.append(name)

        self.exec_order = exec_order[len(inputs):] # Skip inputs
        self.exec_conf  = exec_conf

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        """ Given the model execution order, process a tensor
        Args:
            x (torch.Tensor): Input tensor
        """
        stack = [x]
        X = stack[0]
        for name in self.exec_order:
            index = self.exec_conf[name]['src_index']
            if len(index) != 1 or index[0] != -1:
                # Fetch inbound tensors (otherwise, use previous one)
                X = [stack[i] if i != -1 else X for i in index]
                if len(X) == 1: X = X[0]
            X = getattr(self,name)(X)
            if self.exec_conf[name]['save']:
                stack.append(X)
        return X

