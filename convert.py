#!/usr/bin/env python3
from argparse import RawTextHelpFormatter
import argparse


def _main():
    ap = argparse.ArgumentParser(
        description="Convert models.", formatter_class=RawTextHelpFormatter
    )
    subparsers = ap.add_subparsers(help="Conversion mode", dest="cmd")

    tf2json_parser = subparsers.add_parser("tf2json")
    tf2json_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        metavar="<path_to_file>",
        help="path to Tensorflow source model.",
    )
    tf2json_parser.add_argument(
        "-i",
        "--inputs",
        type=str,
        nargs="+",
        required=False,
        metavar="<name>",
        help="name of input layers.",
    )
    tf2json_parser.add_argument(
        "-o",
        "--outputs",
        type=str,
        nargs="+",
        required=False,
        metavar="<name>",
        help="name of output layers.",
    )
    tf2json_parser.add_argument(
        "-j",
        "--json",
        type=str,
        required=True,
        metavar="<path_to_file>",
        help="path to output JSON file.",
    )
    tf2json_parser.add_argument(
        "-v", "--verbose", required=False, action="store_true", help="verbose mode."
    )

    json2pt_parser = subparsers.add_parser("json2torch")
    json2pt_parser.add_argument(
        "-j",
        "--json",
        type=str,
        required=True,
        metavar="<path_to_file>",
        help="path to input JSON file.",
    )
    json2pt_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        metavar="<path_to_file>",
        help="path to Pytorch destination model.",
    )
    json2pt_parser.add_argument(
        "-v", "--verbose", required=False, action="store_true", help="verbose mode."
    )

    args = ap.parse_args()

    if args.cmd == "tf2json":
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import keras
        from tf2json import to_json

        # Load source
        tf_model = keras.models.load_model(args.model, compile=False)
        if hasattr(tf_model, "model"):  # Load exotic models (e.g. _UserObject)
            tf_model = tf_model.model  # type: ignore
        assert isinstance(
            tf_model, keras.src.engine.functional.Functional
        ), f"Unsupported model format {type(tf_model)}"

        if args.verbose:
            tf_model.summary()

        # Convert model and save
        to_json(
            source=tf_model,
            dest=args.json,
            input_nodes=args.inputs,
            output_nodes=args.outputs,
        )

    elif args.cmd == "json2torch":
        from json2torch import ModelFromJson
        import torch

        # Load model from JSON file
        pt_model = ModelFromJson(args.json)
        if args.verbose:
            print(pt_model)

        # Save to torch format
        torch.save(pt_model, args.model)


if __name__ == "__main__":
    _main()
