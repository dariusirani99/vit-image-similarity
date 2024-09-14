import torch
import torch.nn.functional as F


def compute_similarity(vector_1: torch.Tensor, vector_2: torch.Tensor) -> float:
    """
    Compute the cosine similarity of two vectors, as torch tensors.

    :param vector_1: First vector as a torch tensor.
    :param vector_2: Second vector as a torch tensor.
    :return: Cosine similarity score as a float.
    """
    similarity = float(F.cosine_similarity(vector_1, vector_2))
    return similarity

