# modules/model.py - Exact Original Architecture Compatible with Your Checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple Double Convolution Block (NO residual connections)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Down Block
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Attention Gate
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Up Block with Attention (using bilinear upsampling, NOT ConvTranspose2d)
class Up(nn.Module):
    def __init__(self, decoder_channels, encoder_channels, out_channels):
        super(Up, self).__init__()
        # Use bilinear upsampling (no learnable parameters)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Attention gate - based on error messages, F_g should match decoder_channels
        self.attn = AttentionGate(F_g=decoder_channels, F_l=encoder_channels, F_int=out_channels // 2)
        
        # Convolution after concatenation
        # From error: up1 expects 192 channels (128 decoder + 64 encoder = 192)
        self.conv = DoubleConv(decoder_channels + encoder_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Apply attention
        x2 = self.attn(g=x1, x=x2)
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Final Output Convolution
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Original Attention U-Net Model (Compatible with your pretrained weights)
class AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):  # Changed to n_classes=3
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Architecture that matches your checkpoint
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)

        # Up blocks - dimensions based on error messages
        self.up1 = Up(128, 64, 64)    # decoder=128, encoder=64, out=64
        self.up2 = Up(64, 32, 32)     # decoder=64, encoder=32, out=32  
        self.up3 = Up(32, 16, 16)     # decoder=32, encoder=16, out=16
        
        self.outc = OutConv(16, n_classes)  # 3 classes based on error

    def forward(self, x):
        x1 = self.inc(x)      # [B, 16, H, W]
        x2 = self.down1(x1)   # [B, 32, H/2, W/2]
        x3 = self.down2(x2)   # [B, 64, H/4, W/4]
        x4 = self.down3(x3)   # [B, 128, H/8, W/8]

        x = self.up1(x4, x3)  # [B, 64, H/4, W/4]
        x = self.up2(x, x2)   # [B, 32, H/2, W/2]
        x = self.up3(x, x1)   # [B, 16, H, W]
        logits = self.outc(x) # [B, 3, H, W]

        # For 3-class segmentation, return raw logits (softmax will be applied in loss)
        return logits