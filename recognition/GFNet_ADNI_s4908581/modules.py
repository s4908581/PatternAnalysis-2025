"""
This module defines components of a neural network architecture inspired by GFNet, including:
    - Mlp: A feedforward neural network module.
    - GlobalFilter: A module that applies global filtering using Fourier transforms.
    - Block: A transformer block that combines normalization, global filtering, and MLP.
    - PatchEmbed: A module that converts images into patch embeddings.
    - GFNet: The main model class that integrates all components into a complete architecture for image classification.

The code utilizes PyTorch's nn.Module as the base class and employs standard neural network layers and functions.
"""

import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
import math
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft


class Mlp(nn.Module):
    """
    A simple feedforward neural network module with two linear layers, an activation function and a dropout layer.
    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        out_features (int): Number of output features.
        act_layer (nn.Module): Activation layer (default: GELU).
        drop (float): Dropout rate (default: 0.0).
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        The forward pass of the Mlp module.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after processing.
        """

        # Apply first linear layer, activation, dropout, second linear layer, and dropout
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class GlobalFilter(nn.Module):
    """
    A module that applies global filtering using Fourier transforms.
    Args:
        dim (int): Dimensionality of the input features.
        h (int): Height of the filter (default: 14).
        w (int): Width of the filter (default: 8).
    """

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        # Initialize complex weights for the filter
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        """
        Forward pass of the GlobalFilter module.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            spatial_size (tuple): Spatial dimensions (height, width) of the input.
        Returns:
            torch.Tensor: Output tensor after applying global filtering.
        """

        B, N, C = x.shape
        # Determine spatial dimensions
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        # Convert to float32 for FFT operations
        x = x.to(torch.float32)

        # Apply the 2D FFT to the input tensor
        # This transforms the input to the frequency domain
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight

        # Apply the inverse 2D FFT to transform back to the spatial domain
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x

class Block(nn.Module):
    """
    This module is a transformer block that combines normalization, global filtering, and MLP.
    Args:
        dim (int): Dimensionality of the input features.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension (default: 4.0).
        drop (float): Dropout rate (default: 0.0).
        drop_path (float): Stochastic depth rate (default: 0.0).
        act_layer (nn.Module): Activation layer (default: GELU).
        norm_layer (nn.Module): Normalization layer (default: LayerNorm).
        h (int): Height for the GlobalFilter (default: 14).
        w (int): Width for the GlobalFilter (default: 8).
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        Forward pass of the Block module.
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
        Returns:
            torch.Tensor: Output tensor after processing.
        """

        # The forward pass applies normalization, global filtering, and MLP with residual connections
        # With reference to the original GFNet paper
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x
        # residual = x
        # x = self.norm1(x)
        # x = self.filter(x)
        # x = self.norm2(x)
        # x = self.mlp(x)
        # x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        # return x
        
    

class PatchEmbed(nn.Module):
    """
    This module converts an image into patch embeddings using a convolutional layer.
    Args:
        img_size (int or tuple): Size of the input image.
        patch_size (int or tuple): Size of each patch.
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) # Calculate number of patches in the image
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # Convolutional layer to extract patches and project to embedding dimension
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Patch embeddings of shape (B, N, embed_dim).
        """

        B, C, H, W = x.shape
        
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # Apply convolution to extract patches and reshape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class GFNet(nn.Module):
    """
    This module defines the GFNet architecture for image classification.
    Args:
        img_size (int): Size of the input image (default: 224).
        patch_size (int): Size of each patch (default: 16).
        in_chans (int): Number of input channels (default: 1).
        num_classes (int): Number of output classes (default: 1000).
        embed_dim (int): Embedding dimension (default: 768).
        depth (int): Number of transformer blocks (default: 8).
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension (default: 4.0).
        representation_size (int): Size of the representation layer (default: None).
        uniform_drop (bool): Whether to use uniform drop path rate (default: False).
        drop_rate (float): Dropout rate (default: 0.0).
        drop_path_rate (float): Stochastic depth rate (default: 0.0).
        norm_layer (nn.Module): Normalization layer (default: LayerNorm).
        dropcls (float): Dropout rate for classification head (default: 0.0).
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=1000, embed_dim=768, depth=8,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 dropcls=0):
        
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Patch Embedding
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()

         # Classification head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.final_dropout = nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        """
        Initialize weights for the model.
        """
        
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """
        Forward pass through the feature extraction layers.
        """

        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        """
        Forward pass through the entire network.
        """

        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x
