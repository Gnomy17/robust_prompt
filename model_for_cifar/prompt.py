
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple
class PromptBlock(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=4, in_chans=3, middle_dim=100, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_tokens = ((img_size[0] - patch_size[0])//stride) * ((img_size[1] - patch_size[1])//stride)

        self.conv1 = nn.Conv2d(in_chans, middle_dim, kernel_size=patch_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(middle_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(middle_dim, embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.relu(self.bn1(self.conv1(x))).flatten(2).transpose(1, 2)
        x = self.linear(x)
        return x
    
