# Documentation for Tensorflow -> Torch model converter

This ongoing project aims at converting a given Tensorflow model into a torch.nn.Module counter-part.

## Setup
To install, simply run
```
$ python3 -m pip -e . -r requirements.txt
```
Note that current requirements are:
- pytorch >= 1.09
- tensorflow >= 2.5.0
- numpy >= 1.19.5
- keras~=2.15.0

but the tool may work with earlier versions of Tensorflow and Pytorch.

## Manual
In order to cut dependencies between Tensorflow and Pytorch libraries, model conversion is performed in two steps:
1) Conversion from a Tensorflow model to a JSON file containing the model graph and weights
```
usage: convert.py tf2json [-h] -m <path_to_file> [-i <name> [<name> ...]] [-o <name> [<name> ...]] -j <path_to_file> [-v]

options:
  -h, --help            show this help message and exit
  -m <path_to_file>, --model <path_to_file>
                        path to Tensorflow source model.
  -i <name> [<name> ...], --inputs <name> [<name> ...]
                        name of input layers.
  -o <name> [<name> ...], --outputs <name> [<name> ...]
                        name of output layers.
  -j <path_to_file>, --json <path_to_file>
                        path to output JSON file.
  -v, --verbose         verbose mode.
```
Note: the `--inputs` and `--outputs` options allows to cut out specific parts of the 
execution graph before export, for instance to remove unsupported layers.

2) Conversion from the JSON file to a nn.Module Torch model
```
usage: convert.py json2torch [-h] -j <path_to_file> -m <path_to_file> [-v]

optional arguments:
  -h, --help            show this help message and exit
  -j <path_to_file>, --json <path_to_file>
                        path to input JSON file.
  -m <path_to_file>, --model <path_to_file>
                        path to Pytorch destination model.
  -v, --verbose         verbose mode.
```

## List of supported layers
```
Activation (relu, sigmoid or linear)
Add
Multiply
(Depthwise)Conv2D
Linear
BatchNormalization
(Global)AveragePooling2D
MaxPooling2D
ZeroPadding2D
Softmax (2D only)
Dropout
Concatenate
Reshape
```

## Supporting new layers
The layer conversion is performed in the `get_layer_config` function located in [tf2json.py](https://github.com/romain-xu-darme/tf2torch/blob/main/tf2json.py) 
and the `to_torch_layer` function located in [json2torch.py](https://github.com/romain-xu-darme/tf2torch/blob/main/json2torch.py).
In order to add support for new types of layers:
1) Identify the layer type using `if isinstance(src,...)`
2) Save layer type and configuration in a dict structure:
```
{'type': <layer_type_name>, 'params': <structure_containing_all_parameters>}
```
3) If the Tensorflow source layer does not correspond to a predefined Pytorch module, create a new nn.Module (see [torch_layers.py](https://github.com/romain-xu-darme/tf2torch/blob/main/torch_layers.py))
4) Instantiate the Pytorch module in `to_torch_layer` and copy all relevant parameters from the configuration.

