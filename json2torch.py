import torch
import json
from torch_layers import *


def to_torch_layer(conf: dict) -> nn.Module:
    r"""Given a configuration dictionary, returns corresponding Pytorch layer.

    Args:
        conf: Layer configuration

    Returns:
        Pytorch nn.Module
    """
    # Recover layer type
    type_name = conf["type"]
    params = conf["params"]

    if type_name == "ZeroPad2d":
        left, right, top, bot = (
            params["left"],
            params["right"],
            params["top"],
            params["bot"],
        )
        return torch.nn.ZeroPad2d(padding=(left, right, top, bot))

    # Layers with learnable weights
    if type_name in ["Conv2d", "Dense"]:
        if type_name == "Conv2d":
            if params["stride"][0] > 1 and params["padding"] == "same":
                pt_layer = Conv2dSame(
                    in_channels=params["in_channels"],
                    out_channels=params["out_channels"],
                    kernel_size=params["kernel_size"],
                    stride=params["stride"],
                    padding=params["padding"],
                    bias=params["use_bias"],
                    groups=params["groups"],
                )
            else:
                pt_layer = torch.nn.Conv2d(
                    in_channels=params["in_channels"],
                    out_channels=params["out_channels"],
                    kernel_size=params["kernel_size"],
                    stride=params["stride"],
                    padding=params["padding"],
                    bias=params["use_bias"],
                    groups=params["groups"],
                )
        else:  # Dense
            pt_layer = nn.Linear(
                in_features=params["in_features"],
                out_features=params["out_features"],
                bias=params["use_bias"],
            )

        # Copy weights
        pt_layer.weight = torch.nn.Parameter(torch.Tensor(params["weights"]))
        if pt_layer.bias is not None:
            pt_layer.bias.data = torch.Tensor(params["bias"])
        # Handle activation function
        aname = params["activation"]
        if aname == "linear":
            return pt_layer
        if aname == "relu":
            return torch.nn.Sequential(pt_layer, torch.nn.ReLU())
        if aname == "sigmoid":
            return torch.nn.Sequential(pt_layer, torch.nn.Sigmoid())
        if aname == "swish":
            return torch.nn.Sequential(pt_layer, torch.nn.SiLU())
        assert False, f"Activation {aname} not implemented (yet?)"

    if type_name == "Dropout":
        return nn.Dropout(p=params["p"])

    if type_name == "BatchNorm2d":
        pt_layer = torch.nn.BatchNorm2d(
            num_features=params["num_features"],
            eps=params["eps"],
            momentum=params["momentum"],
        )
        # Copy weights
        pt_layer.gamma = torch.nn.Parameter(torch.Tensor(params["gamma"]))
        pt_layer.weight = torch.nn.Parameter(torch.Tensor(params["gamma"]))
        pt_layer.beta = torch.nn.Parameter(torch.Tensor(params["beta"]))
        pt_layer.bias = torch.nn.Parameter(torch.Tensor(params["beta"]))
        pt_layer._buffers["running_mean"] = torch.Tensor(params["mean"])
        pt_layer._buffers["running_var"] = torch.Tensor(params["var"])
        return pt_layer

    if type_name == "Activation":
        aname = params["name"]
        if aname == "relu":
            return torch.nn.ReLU()
        if aname == "sigmoid":
            return torch.nn.Sigmoid()
        if aname == "linear":
            return torch.nn.Identity()
        if aname == "swish":
            return torch.nn.SiLU()
        assert False, f"Activation {aname} not implemented (yet?)"

    if type_name == "MaxPool2d":
        return torch.nn.MaxPool2d(
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            padding=params["padding"],
        )

    if type_name == "AveragePool2d":
        return torch.nn.AvgPool2d(
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            padding=params["padding"],
        )
    
    if type_name == "GlobalAveragePool2d":
        return GlobalAveragePool2d(
            kernel_size=params["kernel_size"],
        )

    if type_name == "Add":
        return Add()

    if type_name == "Multiply":
        return Multiply()

    if type_name == "Concatenate":
        return Concatenate(axis=params["axis"])

    if type_name == "Reshape":
        return Reshape(shape=params["shape"])

    if type_name == "Softmax2d":
        return ChannelWiseSoftmax()

    assert False, f"Layer {type_name} not implemented (yet?)"


class ModelFromJson(nn.Module):
    def __init__(
        self,
        fpath: str,
    ) -> None:
        """Init model from JSON configuration
        Args:
            fpath (str): Path to JSON file
        """
        super(ModelFromJson, self).__init__()

        # Read configuration file
        with open(fpath, "r") as fin:
            config = json.load(fin)

        # Execution order
        self.exec_order = config["exec"]
        self.exec_conf = {}

        # Layer params
        layers_conf = config["layers"]
        # Set each layer as attribute
        for name in layers_conf:
            # Build layer from configuration
            setattr(self, name, to_torch_layer(layers_conf[name]["params"]))
            # Store execution config
            self.exec_conf[name] = {
                "src": layers_conf[name]["exec"]["src"],
                "save": layers_conf[name]["exec"]["save"],
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given the model execution order, process a tensor
        Args:
            x (torch.Tensor): Input tensor
        """
        stack = [x]
        X = stack[0]
        for name in self.exec_order:
            index = self.exec_conf[name]["src"]
            if len(index) != 1 or index[0] != -1:
                # Fetch inbound tensors (otherwise, use previous one)
                X = [stack[i] if i != -1 else X for i in index]
                if len(X) == 1:
                    X = X[0]
            X = getattr(self, name)(X)
            if self.exec_conf[name]["save"]:
                stack.append(X)
        return X
