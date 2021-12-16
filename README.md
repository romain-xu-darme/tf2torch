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

but the tool may work with earlier versions of Tensorflow and Pytorch.

## Manual
In order to cut dependencies between Tensorflow and Pytorch libraries, model conversion is performed in two steps:
1) Conversion from a Tensorflow model to a JSON file containing the model graph and weights
```
usage: convert.py tf2json [-h] -m <path_to_file> [-l <name>] -j <path_to_file> [-v]

optional arguments:
  -h, --help            show this help message and exit
  -m <path_to_file>, --model <path_to_file>
                        Path to Tensorflow source model.
  -l <name>, --layer <name>
                        Last layer of source model.
  -j <path_to_file>, --json <path_to_file>
                        Path to output JSON file.
  -v, --verbose         Verbose.
```
2) Conversion from the JSON file to a nn.Module Torch model
```
usage: convert.py json2torch [-h] -j <path_to_file> -m <path_to_file> [-v]

optional arguments:
  -h, --help            show this help message and exit
  -j <path_to_file>, --json <path_to_file>
                        Path to input JSON file.
  -m <path_to_file>, --model <path_to_file>
                        Path to Pytorch destination model.
  -v, --verbose         Verbose.
```

## List of supported layers
```
Activation (relu, sigmoid or linear)
Add
Conv2D
BatchNormalization
MaxPooling2D
ZeroPadding2D
Softmax (2D only)
```

## Supporting new layers
The layer conversion is performed in the *tflayer2dict* function located in [tf2json.py](https://github.com/romain-xu-darme/tf2torch/blob/main/tf2json.py) and the *dict2torchlayer* function located in [json2torch.py](https://github.com/romain-xu-darme/tf2torch/blob/main/json2torch.py).
In order to add support for new types of layers:
1) Identify the layer name, which corresponds to the *src* input layer class name in *tflayer2dict*, stripped of all module prefixes. *E.g.* *tensorflow.python.keras.layers.normalization_v2.BatchNormalization* becomes *BatchNormalization*
2) Save layer type and configuration in a dict structure:
```
{'type': <layer_type_name>, 'params': <structure_containing_all_parameters>}
```
3) If the Tensorflow source layer does not correspond to a predefined Pytorch module, create a new nn.Module (see "Add" module provided) in *json2torch.py*
4) Instantiate the Pytorch module in *dict2torchlayer* and copy all relevant parameters from the configuration.

