from srcs.model_architecture import CustomViTModel
from torchvision import transforms
from PIL import Image
from functions import helper_functions
import torch

model = CustomViTModel(img_size=512, in_channels=3, patch_size=16, num_transformer_layers=30,
                       embedding_dim=2048, mlp_size=5186, num_heads=16, attn_dropout=.2, mlp_dropout=.2,
                       embedding_dropout=.2, num_classes=1)

model.load_state_dict(r"")

model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

image_ham_1 = Image.open(r"C:\Users\dariu\Desktop\Python\Python Scripts\PyTorch\Data\food-10\data\train\hamburger\72111.jpg")
image_ham_2 = Image.open(r"C:\Users\dariu\Desktop\Python\Python Scripts\PyTorch\Data\food-10\data\train\hamburger\656458.jpg")
image_other = Image.open(r"C:\Users\dariu\Desktop\Python\Python Scripts\PyTorch\Data\food-10\data\train\pizza\2965.jpg")
tensor_ham1 = transform(image_ham_1).unsqueeze(dim=0)
tensor_ham2 = transform(image_ham_2).unsqueeze(dim=0)
tensor_other = transform(image_other).unsqueeze(dim=0)

with torch.inference_mode():
    results_ham1 = model(tensor_ham1)
    results_ham2 = model(tensor_ham2)
    results_other = model(tensor_other)
    ham1_normalized = helper_functions.normalize_vector(results_ham1)
    ham2_normalized = helper_functions.normalize_vector(results_ham2)
    other_normalized = helper_functions.normalize_vector(results_other)

    # computing similarity
    similarity_hams = helper_functions.compute_similarity(ham1_normalized, ham2_normalized)

    similarity_ham1_other = helper_functions.compute_similarity(ham1_normalized, other_normalized)

    similarity_ham2_other = helper_functions.compute_similarity(ham2_normalized, other_normalized)

    print(f"Similarity between hams is {similarity_hams}")
    print(f"Similarity between ham1 and other is {similarity_ham1_other}")
    print(f"Similarity between ham2 and other is {similarity_ham2_other}")

