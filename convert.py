#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tf2torch import TFConvertedModel
import torch
import numpy as np
from argparse import RawTextHelpFormatter
import argparse

if __name__ == '__main__':

    ap = argparse.ArgumentParser(
        description='Convert tensorflow model to pytorch.',
        formatter_class=RawTextHelpFormatter)
    ap.add_argument('-m','--model', type=str, required=True,
        metavar=('<path_to_file>'),
        help='Path to Tensorflow source model.')
    ap.add_argument('-o','--output', type=str, required=True,
        metavar=('<path_to_file>'),
        help='Path to Pytorch destination model.')
    ap.add_argument('-l','--layer', type=str, required=False,
        metavar=('<name>'),
        help='Last layer of source model.')
    ap.add_argument('-v','--verbose',required=False,action='store_true',
        help='Verbose.')
    args = ap.parse_args()

    # Load source
    tf_model = tf.keras.models.load_model(args.model,compile=False)
    if args.layer is not None:
        lnames = [l.name for l in tf_model.layers]
        if args.layer not in lnames:
            ap.error(f'Invalid layer name {args.layer}. Choices: {lnames}')
        tf_model = tf.keras.models.Model(tf_model.inputs,tf_model.get_layer(args.layer).output)
    if args.verbose: tf_model.summary()

    # Convert model and save
    pt_model = TFConvertedModel(tf_model)
    torch.save(pt_model,args.output)
    if args.verbose: print(pt_model)

    # Sanity check: Process random input through both models
    tf_X = np.random.random([10]+list(tf_model.input_shape[1:]))
    tf_out = tf_model.predict(tf_X)
    loaded_pt_model = torch.load(args.output)
    pt_X = torch.Tensor(np.moveaxis(tf_X,3,1))
    loaded_pt_model.eval()
    pt_out = loaded_pt_model(pt_X).detach().numpy()
    pt_out = np.moveaxis(pt_out,1,-1)
    print(f'Maximum absolute error on 10 inputs: {np.amax(np.abs(pt_out-tf_out))}')

