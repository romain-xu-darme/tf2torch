import torch
import torch.nn as nn
from typing import List, Dict

class ConvertedModel (nn.Module):
    def __init__ (self,
        layers: Dict,
        exec_order: List[str],
        exec_conf : Dict,
    ) -> None:
        """ Init converted model from configuration
        Args:
            layers (dict): Dictionnary containing layer names and functions
            exec_order (list): Execution order during forward pass
            exec_conf (dict): Execution configuration for each layer
        """
        super(ConvertedModel,self).__init__()

        # Set each layer as attribute
        for name in layers:
            setattr(self,name,layers[name])

        # Save execution order and configuration
        self.exec_order = exec_order
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

