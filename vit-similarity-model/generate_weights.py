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
