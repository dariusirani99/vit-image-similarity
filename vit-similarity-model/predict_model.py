import torchvision.transforms
from PIL import Image, ImageDraw, ImageFont
from torch import nn
import torch
import os
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_pil_image
import yaml
import json


def load_config(config_file):
    """
    Loads the config from config file.

    Arguments:
        config_file: The path to the config file in the repository.
    """
    with open(config_file) as file:
        return yaml.safe_load(file)


def get_args_parser(add_help=True):
    """Parse arguments from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ViT Model Prediction", add_help=add_help
    )

    parser.add_argument(
        "--input-file", "--input-file", required=True, type=str, help="Input filepath"
    )
    parser.add_argument(
        "--model-file", "--model-file", required=True, type=str, help="Model pt file"
    )
    parser.add_argument(
        "--output-folder",
        default=(str(os.path.dirname(__file__)) + "/output/"),
        type=str,
        help="File to save output",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device (Use cuda or cpu Default: cpu)",
    )

    return parser


def load_model(args):
    """Load model from checkpoint for evaluation, and store it on device."""
    print("[INFO] Loading model from pt file...")
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    model.heads = nn.Linear(in_features=768, out_features=4)
    model.load_state_dict(state_dict=torch.load(args.model_file, weights_only=True))
    model = model.to(torch.device(args.device))

    model.eval()

    return model


def from_file_to_torch_tensor(input_file, device=None):
    """Read input file and transform it to a torch.tensor."""
    with Image.open(input_file) as im:
        image = im.convert("RGB")
        img_tensor = TF.pil_to_tensor(image)
        if device:
            img_tensor = img_tensor.to(device)
        img_tensor = img_tensor / 255.0

    return img_tensor, image


@torch.no_grad()
def predict(image, model):
    """Takes a torch.tensor image and a model and outputs the detection predictions."""

    image = image.unsqueeze(dim=0)

    model.eval()

    return model(image)


def postprocess(data):
    """
    Postprocesses data with specified threshold.

    :param data: Data to post-process.

    :return: Results, post processed.
    """
    probabilities = []
    indices = []
    results = []

    for result in data:
        probability = torch.softmax(result, dim=0)
        index = torch.argmax(probability)

        # Move tensors to CPU and convert them to numpy
        probabilities.append(probability.cpu().numpy())
        indices.append(index.cpu().item())

    # Convert probabilities to list format
    probabilities = [prob.tolist() for prob in probabilities]
    results.append({"index": indices, "probability": probabilities})
    print(results)

    return results


def plot_and_save_image(
    pil_img: Image, results: list, transforms=None, output_image_path="output_image.png"
):
    """
    Plot the image with classes and probabilities, and save the output image.

    :param pil_img: The pillow image.
    :param results: The list of dictionaries containing detection results.
    :param output_image_path: Path to save the output image with overlays.
    :param transforms: Torchvision transforms, if present.
    """
    classes = ['bolt', 'locating pin', 'nut', 'washer']

    if transforms:
        image_tensor = transforms(pil_img).to(torch.device("cpu"))
        pil_img = to_pil_image(image_tensor)

    draw = ImageDraw.Draw(pil_img)

    # Optional: Load a custom font
    try:
        font = ImageFont.truetype("arial.ttf", 18)  # Ensure this font file is available
    except IOError:
        font = ImageFont.load_default()

    # Process each result and overlay class name and probability
    for idx, result in enumerate(results[0]['index']):
        probability = results[0]['probability'][idx]
        class_name = classes[result]
        prob_text = f"{class_name}: {max(probability) * 100:.2f}%"

        # Choose a position (e.g., top-left corner of the image or stack them)
        position = (10, 10 + idx * 30)

        # Draw text on the image
        draw.text(position, prob_text, fill="red", font=font)

    # Save the image with overlays
    pil_img.save(output_image_path)


def main(args):
    """Make a prediction of balloon detections on a single input image."""
    device = torch.device(args.device)

    print(f"[INFO] Device loaded on {device}...")

    image_processing = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    model = load_model(args)
    img, pil_image = from_file_to_torch_tensor(args.input_file, device=device)

    tensor_image = image_processing(pil_image).to(device)

    outputs = predict(tensor_image, model)

    results = postprocess(outputs)



    print(f"[INFO] results: {results}")

    output_results = args.output_folder + "predict_output.json"
    output_plot = args.output_folder + "predict_image.png"

    with open(output_results, "w") as f:
        json.dump(results, f)

    plot_and_save_image(
        pil_img=pil_image,
        output_image_path=output_plot,
        transforms=image_processing,
        results=results,
    )

    return results


if __name__ == "__main__":
    """Application entry point."""
    args = get_args_parser().parse_args()
    main(args=args)
