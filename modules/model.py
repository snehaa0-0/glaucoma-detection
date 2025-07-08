# modules/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Double Convolution Block with Residual Connection
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_p=0.2):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        return self.double_conv(x) + residual

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

# Up Block with Attention
class Up(nn.Module):
    def __init__(self, decoder_channels, encoder_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, kernel_size=2, stride=2)

        self.attn = AttentionGate(F_g=decoder_channels // 2 if not bilinear else decoder_channels, 
                                F_l=encoder_channels, F_int=out_channels // 2)
        self.conv = DoubleConv(decoder_channels // 2 + encoder_channels if not bilinear else decoder_channels + encoder_channels, 
                             out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x2 = self.attn(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Final Output Convolution
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Attention U-Net Model (Using the original name for compatibility)
class AttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Increased feature capacity
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # Added an extra down level

        self.up1 = Up(1024, 512, 512, bilinear=bilinear)
        self.up2 = Up(512, 256, 256, bilinear=bilinear)
        self.up3 = Up(256, 128, 128, bilinear=bilinear)
        self.up4 = Up(128, 64, 64, bilinear=bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      # [B, 64, H, W]
        x2 = self.down1(x1)   # [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # [B, 1024, H/16, W/16]

        x = self.up1(x5, x4)  # [B, 512, H/8, W/8]
        x = self.up2(x, x3)   # [B, 256, H/4, W/4]
        x = self.up3(x, x2)   # [B, 128, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64, H, W]
        logits = self.outc(x) # [B, n_classes, H, W]

        # Multi-class handling
        if self.n_classes > 1:
            return logits  # Will use CrossEntropyLoss or softmax in training code
        else:
            return torch.sigmoid(logits)