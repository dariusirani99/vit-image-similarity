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
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def get_requirements() -> list:
    """Reads the txt requirements file and downloads the dependencies into the dependencies directory."""
    requirements_path = r"serve/requirements.txt"

    with open(requirements_path) as file:
        requirements = [
            line.strip()
            for line in file
            if not line.startswith("--find-links") and not line.startswith("--no-index")
        ]
    return requirements


def download_dependencies(requirements) -> None:
    """Download the whl dependency files into the serve/dependencies directory."""

    if not os.path.exists("serve/dependencies"):
        os.mkdir("serve/dependencies")

    for req in requirements:
        if req:  # Only process non-empty lines
            download_command = [
                "pip",
                "download",
                "--only-binary=:all:",
                "--dest",
                "serve/dependencies",
                req,
            ]
            logger.info(f"Downloading: {req}")
            subprocess.run(download_command, check=True)


def main():
    """Application entry point."""
    parser = argparse.ArgumentParser()

    command_parsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    command_parsers.add_parser("download")
    args = parser.parse_args()

    if args.command == "download":
        requirements = get_requirements()
        download_dependencies(requirements=requirements)
    else:
        raise ValueError("command unknown")


if __name__ == "__main__":
    """Main execution of script."""
    main()
