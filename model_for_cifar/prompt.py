
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple
class PromptBlock(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.conv1 = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU()
    
    def set_prompt(self, prompt):
        self.prompt = nn.Parameter(prompt.prompt.clone().detach())

