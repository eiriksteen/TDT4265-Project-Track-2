import torch
import torch.nn as nn


class ResBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
        )

        self.merge_conv = nn.Conv3d(in_channels*2, in_channels, 3, 1, 1)
        self.adapt_shape =  nn.Conv3d(in_channels, out_channels, 3, 1, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.BatchNorm3d(out_channels)

    def forward(self, x, block_output=None):

        if block_output is not None:
            x = self.merge_conv(torch.concat((x, block_output), dim=1))

        return self.norm(self.adapt_shape(x) + self.block(x))

class Encoder3D(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.ModuleList([
            ResBlock3D(in_channels, 32),
            nn.Conv3d(32, 64, 3, 2, 1),
            ResBlock3D(64, 64),
            nn.Conv3d(64, 128, 3, 2, 1),
            ResBlock3D(128, 128),
            nn.Conv3d(128, 128, 3, 2, 1),
        ])

    def forward(self, x):

        logits = x
        block_outputs = []

        for layer in self.net:
            logits = layer(logits)
            if isinstance(layer, ResBlock3D):
                block_outputs.append(logits)

        return logits, block_outputs
    
class Decoder3D(nn.Module):

    def __init__(self, out_channels):
        super().__init__()

        self.net = nn.ModuleList([
            nn.ConvTranspose3d(128, 128, 3, 2, 1, output_padding=1),
            ResBlock3D(128, 128),
            nn.ConvTranspose3d(128, 64, 3, 2, 1, output_padding=1),
            ResBlock3D(64, 64),
            nn.ConvTranspose3d(64, 32, 3, 2, 1, output_padding=1),
            ResBlock3D(32, out_channels),
        ])

    def forward(self, x, block_outputs):

        logits = x
        for layer in self.net:
            if isinstance(layer, ResBlock3D):
                logits = layer(logits, block_outputs.pop())
            else:
                logits = layer(logits)

        return logits
    
class UNet3D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.encoder = Encoder3D(in_channels)
        self.decoder = Decoder3D(out_channels)

    def forward(self, x):

        logits, block_outputs = self.encoder(x)
        logits = self.decoder(logits, block_outputs)

        return logits
