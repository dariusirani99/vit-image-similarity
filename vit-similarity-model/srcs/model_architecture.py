import torch
from torch import nn
import torchvision
import yaml


# ViT Classes
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    :param patch_size: Size of patches to convert input image into.
    :param in_channels (int): Number of color channels for the input images.
    :param embedding_dim (int): Size of embedding to turn image into.
    """

    # Initialize the class with appropriate variables
    def __init__(self,
                 patch_size: int,
                 embedding_dim: int,
                 in_channels: int,
                 ):
        self.patch_size = patch_size
        super().__init__()

        # Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,  # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, (f"Input image size must be divisible by patch size, "
                                                         f"image shape: {image_resolution}, patch size: {self.patch_size}")

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # Make sure the output shape has the right order
        return x_flattened.permute(0, 2, 1)


class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""

    # initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim: int,  # Hidden Size D from Table 1 for ViT-Base
                 mlp_size: int,  # MLP size from Table 1 for ViT-Base
                 dropout: float):  # Dropout from Table 3 for ViT-Base
        super().__init__()

        # create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),  # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,  # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim),  # take back to embedding_dim
            nn.Dropout(p=dropout)  # "Dropout, when used, is applied after every dense layer.."
        )

    # create a forward() method to pass the data through the layers
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block with MSA and MLP blocks."""

    # initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim: int,  # Hidden Size D from Table 1 for ViT-Base
                 num_heads: int,  # Heads from Table 1
                 mlp_size: int,  # MLP size from Table 1 for ViT-Base
                 mlp_dropout: float,  # Dropout from Table 3 for ViT-Base
                 attn_dropout: float):  # Dropout for attention layers
        super().__init__()

        # Creating MSA Block
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        # Creating MLP Block
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x  # MSA block with residual connections
        x = self.mlp_block(x) + x  # MLP block with residual connections
        return x


class MultiheadSelfAttentionBlock(nn.Module):
    """
    Creates a multi-head self-attention block ("MSA block" for short).

    :param embedding_dim: The hidden embedding dim 'D'.
    :param num_heads: Chosen number of heads.
    :param attn_dropout: Percent dropout of the layers
    """

    # initialize the class with hyperparameters from Table 1
    def __init__(self,
                 embedding_dim: int,  # Hidden size D from Table 1 for ViT-Base
                 num_heads: int,  # Heads from Table 1 for ViT-Base
                 attn_dropout: float):  # percent dropout
        super().__init__()

        # create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)  # does our batch dimension come first?

    # forward() method to pass the data through the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,  # query embeddings
                                             key=x,  # key embeddings
                                             value=x,  # value embeddings
                                             need_weights=False)  # do we need the weights or just the layer outputs?
        return attn_output


class CustomViTModel(nn.Module):
    def __init__(self,
                 img_size: int,  # Training resolution
                 in_channels: int,  # Number of channels in input image
                 patch_size: int,  # Patch size
                 num_transformer_layers: int,  # total transformer layers
                 embedding_dim: int,  # hidden size D
                 mlp_size: int,  # MLP size from Table 1 for ViT-Base
                 num_heads: int,  # Heads
                 attn_dropout: float,  # Dropout for attention projection
                 mlp_dropout: float,  # Dropout for dense/MLP layers
                 embedding_dropout: float,  # Dropout for patch and position embeddings
                 num_classes: int):
        super().__init__()

        assert img_size % patch_size == 0, (f"Image size must be divisible by patch size, "
                                            f"image size: {img_size}, patch size: {patch_size}.")

        self.number_patches = (img_size * img_size) // patch_size**2

        self.class_embedding_token = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                                  requires_grad=True)

        self.position_embedding = nn.Parameter(data=torch.randn(1, self.number_patches + 1, embedding_dim),
                                               requires_grad=True)

        self.embedding_dropout_layer = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           mlp_dropout=mlp_dropout,
                                                                           attn_dropout=attn_dropout) for _ in
                                                   range(num_transformer_layers)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        class_token = self.class_embedding_token.expand(batch_size, -1,
                                                        -1)

        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout_layer(x)
        x = self.transformer_encoder(x)
        x_vector = x[:, 0]
        x = self.classifier(x[:, 0])
        return x, x_vector


class PreTrainedViT(nn.Module):
    def __init__(self,
                 train_config=None):
        super().__init__()
        if train_config:
            self.config = train_config
        else:
            self.config = self.load_config("config.yml")
        # Loading Device
        if self.config["use_cuda"]:
            print(
                "ATTENTION: The config.yml has been set to 'use_cuda' = True. The device being used is now 'cuda'."
            )
            self.device = torch.device("cuda")
        else:
            print("ATTENTION: Cuda not available. The device being used is now 'cpu'.")
            self.device = torch.device("cpu")

        self.model_base = torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.DEFAULT)

        self.model_base.to(self.device)

    def forward(self, x):
        # Forward pass through the base model (excluding the classification head)
        x = self.model_base(x)
        return x

    @staticmethod
    def load_config(config_file):
        """
        Loads the config from config file.

        Arguments:
            config_file: The path to the config file in the repository.
        """
        with open(config_file) as file:
            return yaml.safe_load(file)
