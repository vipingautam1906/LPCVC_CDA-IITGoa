import numpy as np
import torch

def video_mix(video, target, alpha=1.0):
    batch_size = video.size(0)
    indices = torch.randperm(batch_size).to(video.device)
    
    video_shuffled = video[indices]
    target_shuffled = target[indices]

    lam = np.random.beta(alpha, alpha)
    
    T, H, W = video.size()[2:]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply the mix across all frames in the temporal dimension T
    video[:, :, :, y1:y2, x1:x2] = video_shuffled[:, :, :, y1:y2, x1:x2]
    
    # Adjust lambda to the actual area ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    
    return video, target, target_shuffled, lam

def stackmix(video, target, alpha=1.0):
    B, C, T, H, W = video.size()
    indices = torch.randperm(B).to(video.device)
    video_shuffled = video[indices]
    target_shuffled = target[indices]
    lam = np.random.beta(alpha, alpha)
    cut_idx = int(lam * T)
    video_mixed = video.clone()  # ✅ avoid inplace
    video_mixed[:, :, cut_idx:] = video_shuffled[:, :, cut_idx:]
    lam = cut_idx / T
    return video_mixed, target, target_shuffled, lam