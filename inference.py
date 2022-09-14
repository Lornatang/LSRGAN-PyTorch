# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os

import cv2
import numpy as np
import torch

import config
import imgproc
import model

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main(args):
    # Initialize the super-resolution msrn_model
    msrn_model = model.__dict__[args.model_arch_name](in_channels=args.in_channels, out_channels=args.out_channels)
    msrn_model = msrn_model.to(device=config.device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load the super-resolution msrn_model weights
    checkpoint = torch.load(args.model_weights_path, map_location=lambda storage, loc: storage)
    msrn_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    msrn_model.eval()

    # Read LR image and HR image
    lr_image = cv2.imread(args.inputs_path).astype(np.float32) / 255.0

    # Convert BGR channel image format data to RGB channel image format data
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

    # Convert RGB channel image format data to Tensor channel image format data
    lr_tensor = imgproc.image_to_tensor(lr_image, False, False).unsqueeze_(0)

    # Transfer Tensor channel image format data to CUDA device
    lr_tensor = lr_tensor.to(device=config.device, non_blocking=True)

    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_tensor = msrn_model(lr_tensor)

    # Save image
    sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_path, sr_image)

    print(f"SR image save to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the MSRN model generator super-resolution images.")
    parser.add_argument("--model_arch_name", type=str, default="msrn_x4")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--inputs_path", type=str, default="./figure/comic_lr.png", help="Low-resolution image path.")
    parser.add_argument("--output_path", type=str, default="./figure/comic_sr.png", help="Super-resolution image path.")
    parser.add_argument("--model_weights_path", type=str,
                        default="./results/pretrained_models/MSRN_x4-DIV2K-572bb58f.pth.tar",
                        help="Model weights file path.")
    args = parser.parse_args()

    main(args)
