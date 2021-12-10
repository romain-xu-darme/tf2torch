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
The conversion is pretty straightforward:
```
$./convert.py -h
usage: convert.py [-h] -m <path_to_file> -o <path_to_file> [-l <name>] [-v]

Convert tensorflow model to pytorch.

optional arguments:
  -h, --help            show this help message and exit
  -m <path_to_file>, --model <path_to_file>
                        Path to Tensorflow source model.
  -o <path_to_file>, --output <path_to_file>
                        Path to Pytorch destination model.
  -l <name>, --layer <name>
                        Last layer of source model.
  -v, --verbose         Verbose.
```

## List of supported layers
```
Activation (relu or linear)
Add
Conv2D
BatchNormalization
MaxPooling2D
ZeroPadding2D
```

## Supporting new layers
The layer conversion is performed in the *convert_layer* function located in [layers.py](https://github.com/romain-xu-darme/tf2torch/blob/main/layers.py)
In order to add support for new types of layers:
1) Identify the layer name, which corresponds to the *src* input layer class name, stripped of all module prefixes. *E.g.* *tensorflow.python.keras.layers.normalization_v2.BatchNormalization* becomes *BatchNormalization*
2) If the Tensorflow source layer does not correspond to a predefined Pytorch module, create a new nn.Module (see "Add" module provided in the code)
3) If the Tensorflow layer corresponds to a predefined Pytorch module, instantiate the Pytorch module and copy all relevant parameters from the source layer.

