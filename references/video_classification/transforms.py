import torch
import torch.nn as nn
import torchvision.transforms.functional as F 
import random
import torchvision.transforms as T

class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)

class VideoColorJitter:
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.7):
        self.brightness = [max(0, 1 - brightness), 1 + brightness]
        self.contrast = [max(0, 1 - contrast), 1 + contrast]
        self.saturation = [max(0, 1 - saturation), 1 + saturation]
        self.hue = [-hue, hue]
        self.p = p

    def __call__(self, clip):
        if random.random() > self.p:
            return clip
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            T.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        t_dim, c_dim, h_dim, w_dim = clip.shape
        frames = []
        for i in range(t_dim):
            frame = clip[i]  

            for transform_id in fn_idx:
                if transform_id == 0 and brightness_factor is not None:
                    frame = F.adjust_brightness(frame, brightness_factor)
                elif transform_id == 1 and contrast_factor is not None:
                    frame = F.adjust_contrast(frame, contrast_factor)
                elif transform_id == 2 and saturation_factor is not None:
                    frame = F.adjust_saturation(frame, saturation_factor)
                elif transform_id == 3 and hue_factor is not None:
                    frame = F.adjust_hue(frame.clamp(0, 1), hue_factor)
            frames.append(frame)
        return torch.stack(frames).clamp(0, 1) 