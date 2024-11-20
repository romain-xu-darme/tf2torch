#!/usr/bin/env python3
from argparse import RawTextHelpFormatter
import argparse


def _main():
    ap = argparse.ArgumentParser(description="Convert models.", formatter_class=RawTextHelpFormatter)
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
        required=False,
        metavar="<name>",
        help="name of input layers.",
        action="append",
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
    tf2json_parser.add_argument("-v", "--verbose", required=False, action="store_true", help="verbose mode.")

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
        "-p",
        "--python",
        type=str,
        required=True,
        metavar="<path_to_python_file>",
        help="path to Python destination file.",
    )
    json2pt_parser.add_argument("-v", "--verbose", required=False, action="store_true", help="verbose mode.")
    json2pt_parser.add_argument("-s", "--sanity", required=False, action="store_true", help="sanity check mode.")

    args = ap.parse_args()

    if args.cmd == "tf2json":
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import keras
        from tf2json import to_json

        # Load source
        from stderr import stderr_redirector
        error = None # Store any error to avoid bypassing the finally from stderr_redirector()
        with stderr_redirector():
            try:
                tf_model = keras.models.load_model(args.model, compile=False)
            except Exception as e:
                error = e
        if error is not None:
            raise error

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

        module_name = args.python
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        pt_model.export(args.model, module_name + ".py", "ModelFromJson")

        if args.sanity:
            module = __import__(module_name)
            m = module.ModelFromJson()
            state_dict = torch.load(args.model, weights_only=True)
            m.load_state_dict(state_dict)
            batch_size = 16
            x = []
            for shape in m.input_shapes:
                batch_shape = [
                    batch_size,
                ] + shape
                x.append(torch.rand(batch_shape))
            if len(x) == 1:
                x = x[0]
            m.eval()
            y = m(x)
            print('Sanity check passed.')
            print(f"Output shape: {y.shape}") # TODO: Should we replace the batch size with, e.g., "B"?



if __name__ == "__main__":
    _main()
