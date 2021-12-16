#!/usr/bin/env python3
from argparse import RawTextHelpFormatter
import argparse

def _main():
    ap = argparse.ArgumentParser(
        description='Convert models.',
        formatter_class=RawTextHelpFormatter)
    subparsers = ap.add_subparsers(help='Conversion mode',dest='cmd')

    tf2json_parser = subparsers.add_parser('tf2json')
    tf2json_parser.add_argument('-m','--model', type=str, required=True,
        metavar=('<path_to_file>'),
        help='Path to Tensorflow source model.')
    tf2json_parser.add_argument('-l','--layer', type=str, required=False,
        metavar=('<name>'),
        help='Last layer of source model.')
    tf2json_parser.add_argument('-j','--json', type=str, required=True,
        metavar=('<path_to_file>'),
        help='Path to output JSON file.')
    tf2json_parser.add_argument('-v','--verbose',required=False,action='store_true',
        help='Verbose.')

    json2pt_parser = subparsers.add_parser('json2torch')
    json2pt_parser.add_argument('-j','--json', type=str, required=True,
        metavar=('<path_to_file>'),
        help='Path to input JSON file.')
    json2pt_parser.add_argument('-m','--model', type=str, required=True,
        metavar=('<path_to_file>'),
        help='Path to Pytorch destination model.')
    json2pt_parser.add_argument('-v','--verbose',required=False,action='store_true',
        help='Verbose.')

    args = ap.parse_args()

    if args.cmd == 'tf2json':
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        from tf2json import tfmodel2json

        # Load source
        tf_model = tf.keras.models.load_model(args.model,compile=False)
        if args.layer is not None:
            lnames = [l.name for l in tf_model.layers]
            if args.layer not in lnames:
                ap.error(f'Invalid layer name {args.layer}. Choices: {lnames}')
            tf_model = tf.keras.models.Model(tf_model.inputs,tf_model.get_layer(args.layer).output)
        if args.verbose: tf_model.summary()

        # Convert model and save
        tfmodel2json(tf_model,args.json)

    elif args.cmd == 'json2torch':
        from json2torch import ModelFromJson
        import torch

        # Load model from JSON file
        pt_model = ModelFromJson(args.json)
        if args.verbose: print(pt_model)

        # Save to torch format
        torch.save(pt_model,args.model)

if __name__ == '__main__':
    _main()
