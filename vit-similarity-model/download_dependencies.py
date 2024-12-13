import argparse
import os
import subprocess


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
            print("[INFO] Downloading dependencies...")
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
