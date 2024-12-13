import os

import torch
import torch.nn as nn
from torchvision import models


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="xLenz Image Feature Extractor Weights Creation", add_help=add_help
    )

    parser.add_argument(
        "-od", "--output-dir", default="./weights", type=str, help="path to saved model"
    )

    return parser


def main(args):
    print(args)

    print("Create model...")
    model = models.resnet34(pretrained=True)
    model.fc = nn.Identity()

    print("Save model...")
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "deepfeatures.pth"))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
