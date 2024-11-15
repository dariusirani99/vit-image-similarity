#  Copyright (c) 2024 Cognition Factory
#  All rights reserved.
#
#  Any use, distribution or replication without a written permission
#  is prohibited.
#
#  The name "Cognition Factory" must not be used to endorse or promote
#  products derived from the source code without prior written permission.
#
#  You agree to indemnify, hold harmless and defend Cognition Factory from
#  and against any loss, damage, claims or lawsuits, including attorney's
#  fees that arise or result from your use the software.
#
#  THIS SOFTWARE IS PROVIDED "AS IS" AND "WITH ALL FAULTS", WITHOUT ANY
#  TECHNICAL SUPPORT OR ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
#  BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. ALSO, THERE IS NO
#  WARRANTY OF NON-INFRINGEMENT, TITLE OR QUIET ENJOYMENT. IN NO EVENT
#  SHALL COGNITION FACTORY OR ITS SUPPLIERS BE LIABLE FOR ANY DIRECT,
#  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
#  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
#  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

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
        "model-file/vitsimilaritymodel.pth",
        "--export-path",
        "output",
        "--handler",
        "srcs/handler.py",
        "--extra-files",
        "config/config.yml,"
        "srcs/model_architecture.py",
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

