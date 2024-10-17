import keras.src.engine.functional
import numpy as np
import json
from keras.layers import (
    InputLayer,
    Layer,
    ZeroPadding2D,
    Conv2D,
    DepthwiseConv2D,
    Dropout,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    Concatenate,
    Add,
    Softmax,
    Reshape,
    Multiply,
    Dense,
)


def get_layer_config(src: Layer) -> dict:
    r"""Returns dictionary containing configuration of a Tensorflow layer.

    Args:
        src: Source layer.

    Returns:
        Dictionary of layer configurations.
    """
    if isinstance(src, ZeroPadding2D):
        (top, bot), (left, right) = src.padding
        return {
            "type": "ZeroPad2d",
            "params": {"left": left, "right": right, "top": top, "bot": bot},
        }

    if isinstance(src, Conv2D):
        params = {
            "in_channels": src.input_shape[-1],
            "out_channels": src.output_shape[-1],
            "kernel_size": src.kernel_size,
            "groups": 1,
            "stride": src.strides,
            "padding": src.padding,
            "use_bias": src.use_bias,
            "activation": src.activation.__name__,
            "weights": np.moveaxis(np.moveaxis(src.weights[0], -1, 0), -1, 1),
        }
        if src.bias is not None:
            params["bias"] = src.weights[1].numpy()
        return {"type": "Conv2d", "params": params}

    if isinstance(src, DepthwiseConv2D):
        params = {
            "in_channels": src.input_shape[-1],
            "out_channels": src.output_shape[-1],
            "groups": int(src.output_shape[-1] / src.depth_multiplier),
            "kernel_size": src.kernel_size,
            "stride": src.strides,
            "padding": src.padding,
            "use_bias": src.use_bias,
            "activation": src.activation.__name__,
            # Swap axes to match Pytorch Conv2D group mode
            "weights": np.moveaxis(np.moveaxis(src.weights[0], -1, 0), -1, 0),
        }
        if src.bias is not None:
            params["bias"] = src.weights[1].numpy()
        return {"type": "Conv2d", "params": params}

    if isinstance(src, Dropout):
        return {
            "type": "Dropout",
            "params": {
                "p": src.rate,
            },
        }

    if isinstance(src, BatchNormalization):
        params = {
            "num_features": src.gamma.shape[0],
            "eps": src.epsilon,
            "momentum": src.momentum,
            "gamma": src.gamma.numpy(),
            "beta": src.beta.numpy(),
            "mean": src.moving_mean.numpy(),
            "var": src.moving_variance.numpy(),
        }
        return {"type": "BatchNorm2d", "params": params}

    if isinstance(src, Activation):
        return {"type": "Activation", "params": {"name": src.activation.__name__}}

    if isinstance(src, MaxPooling2D):
        assert src.padding == "valid", "'Same' padding not supported (yet?)"
        return {
            "type": "MaxPool2d",
            "params": {
                "padding": 0,
                "kernel_size": src.pool_size,
                "stride": src.strides,
            },
        }

    if isinstance(src, AveragePooling2D):
        assert src.padding == "valid", "'Same' padding not supported (yet?)"
        return {
            "type": "AveragePool2d",
            "params": {
                "padding": 0,
                "kernel_size": src.pool_size,
                "stride": src.strides,
            },
        }

    if isinstance(src, GlobalAveragePooling2D):
        return {
            "type": "GlobalAveragePool2d",
            "params": {"kernel_size": src.input_shape[1:3]},
        }

    if isinstance(src, Concatenate):
        # Permute axis to match channel-first mode
        # i.e. N x ... x C will become N x C x ....
        ndim = len(src.input_shape[0])
        assert -1 <= src.axis <= ndim, f"Invalid concatenation axis {src.axis} ({ndim} dimensions)"
        match src.axis:
            case -1:
                axis = 1
            case 0:
                axis = 0
            case _:
                axis = 1 + (src.axis + 1) % ndim
        return {
            "type": "Concatenate",
            "params": {
                "axis": axis,
            },
        }

    if isinstance(src, Reshape):
        # Permute axis to match channel-first mode
        return {
            "type": "Reshape",
            "params": {"shape": (src.target_shape[-1],) + src.target_shape[:-1]},
        }

    if isinstance(src, Add):
        return {"type": "Add", "params": None}

    if isinstance(src, Multiply):
        return {"type": "Multiply", "params": None}

    if isinstance(src, Softmax):
        # Assume the following TF tensor format: N x H x W x C
        assert src.axis == [
            1,
            2,
        ], f"Softmax axis support not implemented (yet?): {src.axis}"
        return {"type": "Softmax2d", "params": None}

    if isinstance(src, Dense):
        params = {
            "in_features": src.input_shape[-1],
            "out_features": src.units,
            "use_bias": src.use_bias,
            "activation": src.activation.__name__,
            "weights": np.moveaxis(np.moveaxis(src.weights[0], -1, 0), -1, 1),
        }
        if src.bias is not None:
            params["bias"] = src.weights[1].numpy()
        return {"type": "Dense", "params": params}

    if src.__class__.__name__ == "MelSpecLayerSimple":
        params = { k: src._config[k] for k in [
                "sample_rate",
                "frame_step",
                "frame_length",
                "fmin",
                "fmax",
        ]}
        params["spec_shape"] = src._config["spec_shape"]
        return {"type": "MelSpecLayerSimple", "params": params}


    assert False, f"Layer {src.__class__.__name__.split('.')[-1]} not implemented (yet?)"


class NumpyEncoder(json.JSONEncoder):
    r"""JSON encoder converting numpy arrays to lists before serialization."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def to_json(
    source: keras.src.engine.functional.Functional,
    dest: str,
    input_nodes: list | None = None,
    output_nodes: list | None = None,
) -> None:
    r"""Converts a Keras Functional model to a JSON file.

    Args:
        source: Source model.
        dest: Path to JSON file.
        input_nodes: List of input nodes. Default: None.
        output_nodes: List of output nodes. Default: None.
    """
    def rename(name):
        return name.replace("-", "_")
    ##############################
    # Build graph
    ##############################
    inputs = []
    graph = {}

    # Select relevant nodes selection (similar to model.summary)
    relevant_nodes = []
    for v in source._nodes_by_depth.values():
        relevant_nodes += v

    # Find name of InputLayers
    input_layers = [layer.name for layer in source.layers if isinstance(layer, InputLayer)]

    # Find all inbound nodes for all layers
    for layer in source.layers:
        name = layer.name

        if name in input_layers:
            continue

        graph[name] = {"inbounds": [], "outbounds": []}
        for node in layer._inbound_nodes:  # type: ignore
            if node not in relevant_nodes:
                continue
            for inbound_layer, _, _, _ in node.iterate_inbound():
                if inbound_layer.name not in input_layers:
                    graph[name]["inbounds"].append(inbound_layer.name)
        if not graph[name]["inbounds"]:
            inputs.append(name)

    # For each node, find outbound nodes
    for name in graph:
        for inbound in graph[name]["inbounds"]:
            graph[inbound]["outbounds"].append(name)

    if output_nodes is None:
        output_nodes = [node_name for node_name, conf in graph.items() if not conf["outbounds"]]
    else:
        for output_node in output_nodes:
            if output_node not in graph:
                raise ValueError(f"Output node {output_node} not present in graph. Possible typo?")
    if input_nodes is None:
        input_nodes = inputs
    else:
        for input_node in input_nodes:
            if input_node not in graph:
                raise ValueError(f"Input node {input_node} not present in graph. Possible typo?")

    # Graph consistency check: are all output nodes computable using input nodes?
    known_computable = set()
    def computable(node_name: str) -> bool:
        if node_name in known_computable:
            return True
        if node_name in input_nodes:
            known_computable.add(node_name)
            return True
        if not graph[node_name]["inbounds"]:
            return False
        for inbound in graph[node_name]["inbounds"]:
            if not computable(inbound):
                return False
        known_computable.add(node_name)
        return True

    for output_node in output_nodes:
        if not computable(output_node):
            raise ValueError(
                "Graph inconsistency detected. "
                f"Output node {output_node} cannot be computed from selected inputs {input_nodes}."
            )
    del known_computable

    # Remove nodes on paths leading to specific inputs
    if input_nodes is not None:
        inputs = input_nodes

        def del_upward(node_name: str):
            for inbound in graph[node_name]["inbounds"]:
                if graph.get(inbound) is not None:
                    del_upward(inbound)
                    # Remove it as an input for OTHER outbound nodes
                    for outbound in graph[inbound]["outbounds"]:
                        if outbound != node_name:
                            graph[outbound]["inbounds"].remove(inbound)
                    print(f"Deleting node {inbound}")
                    del graph[inbound]
            graph[node_name]["inbounds"] = []

        for name in inputs:
            del_upward(name)

    # Remove nodes on paths leading from specific outputs
    if output_nodes is not None:

        def del_downward(node_name: str):
            for outbound in graph[node_name]["outbounds"]:
                if graph.get(outbound) is not None:
                    del_downward(outbound)
                    # Remove it as an output for OTHER incoming nodes
                    for inbound in graph[outbound]["inbounds"]:
                        if inbound != node_name:
                            graph[inbound]["outbounds"].remove(outbound)
                    print(f"Deleting node {outbound}")
                    del graph[outbound]
            graph[node_name]["outbounds"] = []

        for name in output_nodes:
            del_downward(name)

    # Recover layer information
    layers_conf = {}
    for layer in source.layers:
        if layer.name in graph:
            layers_conf[layer.name] = {"params": get_layer_config(layer)}

    # Find execution order recursively
    exec_order = []

    def add_children(node_name: str):
        if node_name in exec_order:
            return
        for inbound in graph[node_name]["inbounds"]:
            # Missing inbound node, can't execute this node yet
            if inbound not in exec_order:
                return
        # Inbound nodes already in exec_order list
        exec_order.append(node_name)
        for outbound in graph[node_name]["outbounds"]:
            add_children(outbound)
        return

    for name in inputs:
        add_children(name)

    # Compute index of inbound tensors
    stack = inputs.copy()
    stack = ['input']
    for idx, name in enumerate(exec_order):
        exec_conf = {"src": [], "save": False}
        if name in inputs:
            exec_conf["src"].append(0)
        for inbound in graph[name]["inbounds"]:
            # Simple case: use previous tensor
            if exec_order[idx - 1] == inbound:
                exec_conf["src"].append(-1)
            # Or find index in stack of tensors
            else:
                exec_conf["src"].append(stack.index(inbound))

        # If tensor value is not used straight away, save it
        num_outbounds = len(graph[name]["outbounds"])
        if (num_outbounds > 1) or (num_outbounds == 1 and exec_order[idx + 1] != graph[name]["outbounds"][0]):
            exec_conf["save"] = True
            stack.append(name)
        layers_conf[name]["exec"] = exec_conf

    input_shapes = []
    for name in inputs:
        input_shape = list(source.get_layer(name).input_shape[1:])  # type: ignore
        # Channel first
        input_shape = [input_shape[-1]] + input_shape[:-1]
        input_shapes.append(input_shape)

    # Save to JSON
    with open(dest, "w") as fout:
        layers_conf = { rename(k):v for (k,v) in layers_conf.items() }
        exec_order = [rename(name) for name in exec_order]
        json.dump(
            {"input_shapes": input_shapes, "layers": layers_conf, "exec": exec_order},
            fout,
            cls=NumpyEncoder,
        )
