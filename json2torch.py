import torch
import json
from torch_layers import *


def to_torch_layer(conf: dict, name: str) -> tuple[nn.Module, set[str], list[str]]:
    r"""Given a configuration dictionary, returns corresponding Pytorch layer.

    Args:
        conf: Layer configuration
        name: Name of the layer

    Returns:
        Pytorch nn.Module
        Set of necessary imports
        List of python lines necessary to generate the model
    """
    # Recover layer type
    type_name = conf["type"]
    params = conf["params"]

    def cleanify_string(lines: str) -> str:
        r"""Turns a string with end-of-lines and whitespaces at the beginning of each line
          into a single line without the whitespaces.

        Args:
            lines: String to cleanify

        Returns:
            Cleanified version of lines.
        """
        return ' '.join([ p.lstrip() for p in lines.split("\n")])

    def int_or_keyword(input: str) -> str:
        r"""Takes a string that is either a keyword or an int,
        and returns that string surrounded with quotes if it is a string.

        Args:
            input: the string

        Returns:
            the string, or the string with quotes around.
        """
        if input.isdigit():
            return input
        return f"\"{input}\""

    if type_name == "ZeroPad2d":
        left, right, top, bot = (
            params["left"],
            params["right"],
            params["top"],
            params["bot"],
        )
        return (
            nn.ZeroPad2d(padding=(left, right, top, bot)),
            set(),
            [f"{name} = nn.ZeroPad2d(padding=({left}, {right}, {top}, {bot}))"]
        )

    # Layers with learnable weights
    if type_name in ["Conv2d", "Dense"]:
        imports = set()
        if type_name == "Conv2d":
            if params["stride"][0] > 1 and params["padding"] == "same":
                pt_layer, pt_descr = Conv2dSame(
                    in_channels=params["in_channels"],
                    out_channels=params["out_channels"],
                    kernel_size=params["kernel_size"],
                    stride=params["stride"],
                    padding=params["padding"],
                    bias=params["use_bias"],
                    groups=params["groups"],
                ), f"""Conv2dSame(
                    in_channels={params["in_channels"]},
                    out_channels={params["out_channels"]},
                    kernel_size={params["kernel_size"]},
                    stride={params["stride"]},
                    padding={int_or_keyword(params["padding"])},
                    bias={params["use_bias"]},
                    groups={params["groups"]},
                )"""
                imports.add("from torch_layers import Conv2dSame")
            else:
                pt_layer, pt_descr = nn.Conv2d(
                    in_channels=params["in_channels"],
                    out_channels=params["out_channels"],
                    kernel_size=params["kernel_size"],
                    stride=params["stride"],
                    padding=params["padding"],
                    bias=params["use_bias"],
                    groups=params["groups"],
                ), f"""nn.Conv2d(
                    in_channels={params["in_channels"]},
                    out_channels={params["out_channels"]},
                    kernel_size={params["kernel_size"]},
                    stride={params["stride"]},
                    padding={int_or_keyword(params["padding"])},
                    bias={params["use_bias"]},
                    groups={params["groups"]},
                )"""
        else:  # Dense
            pt_layer, pt_descr = nn.Linear(
                in_features=params["in_features"],
                out_features=params["out_features"],
                bias=params["use_bias"],
            ), f"""nn.Linear(
                in_features={params["in_features"]},
                out_features={params["out_features"]},
                bias={params["use_bias"]},
            )"""
        pt_descr = f"pt_layer = {cleanify_string(pt_descr)}"

        # Copy weights
        pt_layer.weight = nn.Parameter(torch.Tensor(params["weights"]))
        if pt_layer.bias is not None:
            pt_layer.bias.data = torch.Tensor(params["bias"])
        # Handle activation function
        aname = params["activation"]
        if aname == "linear":
            layer = pt_layer
            actual_descr = "pt_layer"
        elif aname == "relu":
            layer = nn.Sequential(pt_layer, nn.ReLU())
            actual_descr = "nn.Sequential(pt_layer, nn.ReLU())"
        elif aname == "sigmoid":
            layer = nn.Sequential(pt_layer, nn.Sigmoid())
            actual_descr = "nn.Sequential(pt_layer, nn.Sigmoid())"
        elif aname == "swish":
            layer = nn.Sequential(pt_layer, nn.SiLU())
            actual_descr = "nn.Sequential(pt_layer, nn.SiLU())"
        else:
            assert False, f"Activation {aname} not implemented (yet?)"
        return layer, imports, [pt_descr, f"{name} = {actual_descr}"]

    if type_name == "Dropout":
        return nn.Dropout(p=params["p"]), set(), [f"""{name} = nn.Dropout(p={params["p"]})"""]

    if type_name == "BatchNorm2d":
        pt_layer = nn.BatchNorm2d(
            num_features=params["num_features"],
            eps=params["eps"],
            momentum=params["momentum"],
        )
        descr = f"""nn.BatchNorm2d(
            num_features={params["num_features"]},
            eps={params["eps"]},
            momentum={params["momentum"]},
        )
        """
        # Copy weights
        pt_layer.gamma = nn.Parameter(torch.Tensor(params["gamma"]))
        pt_layer.weight = nn.Parameter(torch.Tensor(params["gamma"]))
        pt_layer.beta = nn.Parameter(torch.Tensor(params["beta"]))
        pt_layer.bias = nn.Parameter(torch.Tensor(params["beta"]))
        pt_layer._buffers["running_mean"] = torch.Tensor(params["mean"])
        pt_layer._buffers["running_var"] = torch.Tensor(params["var"])
        return pt_layer, set(), [
            f"{name} = {cleanify_string(descr)}",
            f"""{name}.gamma = nn.Parameter(torch.Tensor({params["gamma"]}))""",
            f"""{name}.beta = nn.Parameter(torch.Tensor({params["beta"]}))""",
        ]

    if type_name == "Activation":
        aname = params["name"]
        if aname == "relu":
            return nn.ReLU(), set(), [f"{name} = nn.ReLU()"]
        if aname == "sigmoid":
            return nn.Sigmoid(), set(), [f"{name} = nn.Sigmoid()"]
        if aname == "linear":
            return nn.Identity(), set(), [f"{name} = nn.Identity()"]
        if aname == "swish":
            return nn.SiLU(), set(), [f"{name} = nn.SiLU()"]
        assert False, f"Activation {aname} not implemented (yet?)"

    if type_name == "MaxPool2d":
        descr = f"""nn.MaxPool2d(
            kernel_size={params["kernel_size"]},
            stride={params["stride"]},
            padding={params["padding"]},
        )"""
        return nn.MaxPool2d(
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            padding=params["padding"],
        ), set(), [f"{name} = {cleanify_string(descr)}"]

    if type_name == "AveragePool2d":
        descr = f"""nn.AvgPool2d(
            kernel_size={params["kernel_size"]},
            stride={params["stride"]},
            padding={params["padding"]},
        )"""
        return nn.AvgPool2d(
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            padding=params["padding"],
        ), set(), [f"{name} = {cleanify_string(descr)}"]

    if type_name == "GlobalAveragePool2d":
        descr = f"""GlobalAveragePool2d(
            kernel_size={params["kernel_size"]},
        )"""
        return (
            GlobalAveragePool2d(kernel_size=params["kernel_size"],),
            set(["from torch_layers import GlobalAveragePool2d"]),
            [f"""{name} = {cleanify_string(descr)}"""]
        )

    if type_name == "Add":
        return Add(), set(["from torch_layers import Add"]), [f"{name} = Add()"]

    if type_name == "Multiply":
        return Multiply(), set(["from torch_layers import Multiply"]), [f"{name} = Multiply()"]

    if type_name == "Concatenate":
        return (
            Concatenate(axis=params["axis"]),
            set(["from torch_layers import Concatenate"]),
            [f"""{name} = Concatenate(axis={params["axis"]})"""]
        )

    if type_name == "Reshape":
        return (
            Reshape(shape=params["shape"]),
            set(["from torch_layers import Reshape"]),
            [f"""{name} = Reshape(shape={params["shape"]})"""]
        )

    if type_name == "Softmax2d":
        return (
            ChannelWiseSoftmax(),
            set(["from torch_layers import Softmax2d"]),
            [f"{name} = ChannelWiseSoftmax()"]
        )

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

        # Input shapes
        self.input_shapes = config["input_shapes"]

        # Execution order
        self.exec_order = config["exec"]
        self.exec_conf = {}

        self.imports = set(["import torch", "import torch.nn as nn"])
        self.layer_descr = []
        # Layer params
        layers_conf = config["layers"]
        # Set each layer as attribute
        for name in layers_conf:
            # Build layer from configuration
            layer, imports, layer_descrs = to_torch_layer(layers_conf[name]["params"], f"self.{name}")
            self.imports.update(imports)
            self.layer_descr.extend(layer_descrs)
            setattr(self, name, layer)
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

    def export(self, state_dict_filename, model_filename, classname="ModelFromJson"):
        torch.save(self.state_dict(), state_dict_filename)

        with open(model_filename, "w") as out:
            class Printer: # Used to write python code
                def __init__(self, out) -> None:
                    self.out = out
                    self.indent = 0
                def inc(self): # increases indentation
                    self.indent += 1
                def dec(self): # decreases indentation
                    self.indent -= 1
                def __call__(self, line: str|list[str]): # prints the line(s)
                    if isinstance(line, str):
                        self.out.write(f"""{"    "*self.indent}{line}\n""")
                    else:
                        for l in line:
                            self(l)

            pr = Printer(out)
            pr(self.imports)
            pr("")
            pr(f"class {classname}(nn.Module):")
            pr.inc()

            # __init__
            pr("def __init__(self):")
            pr.inc()
            pr(f"super({classname}, self).__init__()")
            pr(f"self.input_shapes = {self.input_shapes}")
            for line in self.layer_descr:
                pr(line)
            pr.dec()

            pr("")

            # forward
            pr("def forward(self, X: torch.Tensor) -> torch.Tensor:")
            pr.inc()
            current_index = 0
            pr("X_0 = X")
            for name in self.exec_order:
                indices = self.exec_conf[name]["src"]
                indices = [ "X" if i == -1 else f"X_{i}" for i in indices ]
                params = indices[0] if len(indices) == 1 else f"""[{",".join(indices)}]"""
                pr(f"""X = self.{name}({params})""")
                if self.exec_conf[name]["save"]:
                    current_index += 1
                    pr(f"""X_{current_index} = X""")
            pr(f"return X")
            pr.dec()

            pr.dec()
