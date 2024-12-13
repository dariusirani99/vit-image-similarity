import argparse
import os
import subprocess
from pathlib import Path


def get_version() -> str:
    """
    Reads the semver version from the version file (VERSION) in the current directory.

    :return:  The semver version as a string
    """
    with open("VERSION") as file:
        version = file.read().rstrip()
        return version


def get_args_parser(add_help=True):
    """Gets parser command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="xLenz Facility Extractor Model Archiving", add_help=add_help
    )

    parser.add_argument(
        "-imp",
        "--input-model-path",
        required=True,
        type=str,
        help="path to model checkpoint",
    )
    parser.add_argument(
        "-od", "--output-dir", required=True, type=str, help="path to saved model"
    )

    return parser


def create_mar_archive() -> None:
    """Created the mar archive and the mar_folder storage (if not exists)."""
    model_version = get_version()

    if not os.path.exists("output"):
        os.mkdir("output")

    archiver_command = [
        "torch-model-archiver",
        "--model-name",
        "vitsimilaritymodel",
        "--version",
        model_version,
        "--serialized-file",
        "model-file/vitsimilaritymodel.pt",
        "--export-path",
        "output",
        "--handler",
        "srcs/handler.py",
        "--extra-files",
        "config/config.yml",
        "--force",
    ]

    subprocess.run(archiver_command, check=True)

    my_file = Path("output/vitsimilaritymodel.mar")
    my_file.rename(my_file.with_name(f"{my_file.stem}-{model_version}.mar"))


def main():
    """Application entry point."""
    parser = argparse.ArgumentParser()

    command_parsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    command_parsers.add_parser("build")
    args = parser.parse_args()

    if args.command == "build":
        create_mar_archive()
    else:
        raise ValueError("command unknown")


if __name__ == "__main__":
    main()

