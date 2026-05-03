import torch
from torch import nn

# class SE3D(nn.Module):
#     def __init__(self, channels, reduction=8):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _, _ = x.shape
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1, 1)
#         return x * y

# class TemporalAttention(nn.Module):
#     def __init__(self, channels, reduction=4):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Conv1d(channels, channels // reduction, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(channels // reduction, channels, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # x: B, C, T, H, W
#         t_feat = x.mean(dim=[3, 4])  # B, C, T
#         attn = self.net(t_feat)      # B, C, T
#         attn = attn.unsqueeze(-1).unsqueeze(-1)
#         return x * (1 + attn)


# class MotionModule(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Conv3d(
#             channels, channels,
#             kernel_size=(3, 1, 1),
#             padding=(1, 0, 0),
#             groups=channels,
#             bias=False
#         )
#         self.bn = nn.BatchNorm3d(channels)

#     def forward(self, x):        
#         diff = x[:, :, 1:] - x[:, :, :-1]
#         last = diff[:, :, -1:]
#         diff = torch.cat([diff, last], dim=2)  # always executed, no branch
#         out = self.bn(self.conv(diff))
#         return x + out
    
# class SEWrapper(nn.Module):
#     def __init__(self, block):
#         super().__init__()
#         self.block = block

#         channels = None
#         for m in reversed(list(block.modules())):
#             if isinstance(m, nn.Conv3d):
#                 channels = m.out_channels
#                 break

#         if channels is None:
#             raise ValueError("Could not find Conv3d in block")

#         self.se = SE3D(channels)

#     def forward(self, x):
#         out = self.block(x)
#         out = self.se(out)
#         return out

# def apply_se_to_model(model):
#     for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
#         layer = getattr(model, layer_name)
#         new_blocks = []
#         for block in layer:
#             new_blocks.append(SEWrapper(block))
#         setattr(model, layer_name, nn.Sequential(*new_blocks))
#     return model

# class R2Plus1DWithAttention(nn.Module):
#     def __init__(self, base_model, num_classes):
#         super().__init__()

#         # Backbone
#         self.stem = base_model.stem
#         self.layer1 = base_model.layer1
#         self.layer2 = base_model.layer2
#         self.layer3 = base_model.layer3
#         self.layer4 = base_model.layer4
#         self.temporal_attn_mid = TemporalAttention(256)
#         self.temporal_attn_late = TemporalAttention(512)
#         self.motion = MotionModule(512)

#         self.pool = base_model.avgpool
#         self.dropout = nn.Dropout(p=0.3)
#         self.fc = nn.Linear(base_model.fc.in_features, num_classes)

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.temporal_attn_mid(x)
#         x = self.layer4(x)
#         x = self.temporal_attn_late(x)
#         x = self.motion(x)
#         x = self.pool(x)
#         x = x.flatten(1)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x


'''
after profling 
'''
# class SE3D(nn.Module):
#     def __init__(self, channels, reduction=32):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _, _ = x.shape
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1, 1)
#         return x * y

# class TemporalAttention(nn.Module):
#     def __init__(self, channels, reduction=4):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Conv1d(channels, channels // reduction, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(channels // reduction, channels, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # x: B, C, T, H, W
#         t_feat = x.mean(dim=[3, 4])  # B, C, T
#         attn = self.net(t_feat)      # B, C, T
#         attn = attn.unsqueeze(-1).unsqueeze(-1)
#         return x * (1 + attn)
    
# class SEWrapper(nn.Module):
#     def __init__(self, block):
#         super().__init__()
#         self.block = block

#         channels = None
#         for m in reversed(list(block.modules())):
#             if isinstance(m, nn.Conv3d):
#                 channels = m.out_channels
#                 break

#         if channels is None:
#             raise ValueError("Could not find Conv3d in block")

#         self.se = SE3D(channels)

#     def forward(self, x):
#         out = self.block(x)
#         out = self.se(out)
#         return out

# def apply_se_to_model(model):
#     for layer_name in ["layer3", "layer4"]:
#         layer = getattr(model, layer_name)
#         new_blocks = []
#         for block in layer:
#             new_blocks.append(SEWrapper(block))
#         setattr(model, layer_name, nn.Sequential(*new_blocks))
#     return model

# class R2Plus1DWithAttention(nn.Module):
#     def __init__(self, base_model, num_classes):
#         super().__init__()

#         # Backbone
#         self.stem = base_model.stem
#         self.layer1 = base_model.layer1
#         self.layer2 = base_model.layer2
#         self.layer3 = base_model.layer3
#         self.layer4 = base_model.layer4
#         self.temporal_attn_mid = TemporalAttention(256)

#         self.pool = base_model.avgpool
#         self.dropout = nn.Dropout(p=0.3)
#         self.fc = nn.Linear(base_model.fc.in_features, num_classes)

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.temporal_attn_mid(x)
#         x = self.layer4(x)
#         x = self.pool(x)
#         x = x.flatten(1)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x

'''
profile 3 vanila architecture with dropout only
checkpoints = "model_checkpoints_with_attention_3clip"
'''

# def apply_se_to_model(model):
#     return model

# class R2Plus1DWithAttention(nn.Module):
#     def __init__(self, base_model, num_classes):
#         super().__init__()
#         self.stem = base_model.stem
#         self.layer1 = base_model.layer1
#         self.layer2 = base_model.layer2
#         self.layer3 = base_model.layer3
#         self.layer4 = base_model.layer4

#         self.pool = base_model.avgpool
#         self.dropout = nn.Dropout(p=0.3)
#         self.fc = nn.Linear(base_model.fc.in_features, num_classes)

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.pool(x)
#         x = x.flatten(1)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x

'''
p4
checkpoints "model_checkpoints_p4" clip len 16
'''
def apply_se_to_model(model):
    return model

class R2Plus1DWithAttention(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.stem = base_model.stem
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.pool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x