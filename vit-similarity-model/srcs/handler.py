import io

import torch
import yaml
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
import base64
import torch.nn.functional as F


class CustomHandler(BaseHandler):

    def __init__(self):
        super().__init__()
        self.config = self.load_config("config.yml")
        self.transform_composed = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.use_cuda = self.config["use_cuda"]
        if self.use_cuda:
            self.torch_device = torch.device("cuda")
        else:
            self.torch_device = torch.device("cpu")

        self.initialized = False

    @staticmethod
    def load_config(config_file):
        with open(config_file) as file:
            return yaml.safe_load(file)

    @staticmethod
    def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the vector for better similarity results.

        :param vector: Input vector as a tensor.
        :return: vector normalized.
        """
        vector = F.normalize(vector, p=2, dim=1)
        return vector

    def initialize(self, context):
        self.model = self.load_model()
        self.model.eval()
        self.initialized = True

    def load_model(self):
        """Loading the model from the .pt model file."""
        model = torch.jit.load(self.config["model"]["model_path"])

        return model

    def preprocess_one_image(self, req):
        """Process one single image."""
        # get image from the request
        image = req.get("data")
        if image is None:
            image = req.get("body")
        # create a stream from the encoded image
        image = Image.open(io.BytesIO(image))
        # Convert image to RGB
        image = image.convert("RGB")
        image = self.transform_composed(image)
        image = image.to(self.torch_device)
        # add batch dim
        image = image.unsqueeze(0)
        return image

    def preprocess(self, requests):
        """Process all the images from the requests and batch them in a Tensor, if there are multiple requests."""
        images = []
        for row in requests:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray, convert it to bytes
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                mode_accepted = "RGB"
                if image.mode != mode_accepted:
                    image = image.convert(mode_accepted)
                image = self.transform_composed(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def postprocess(self, inference_output):
        """Returning list of features from model."""
        return [self.normalize_vector(inference_output).tolist()]

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.

        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.

        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        with torch.jit.optimized_execution(False):
            with torch.no_grad():
                results = self.model(data, *args, **kwargs)
        return results
