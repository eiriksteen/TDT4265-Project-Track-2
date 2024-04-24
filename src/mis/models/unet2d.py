import torch.nn as nn


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResnetBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, stride=1)
        )

        self.adapt_shape = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, block_output=None):

        if block_output is not None:
            x += block_output

        logits = self.block(x)
        out = self.norm(logits + self.adapt_shape(x))

        return out


class Encoder(nn.Module):

    def __init__(self, in_channels: int):
        super(Encoder, self).__init__()

        self.network = nn.ModuleList([
            ResnetBlock(in_channels, 64),
            nn.Conv2d(64, 128, 3, 2, 1),
            ResnetBlock(128, 128),
            nn.Conv2d(128, 256, 3, 2, 1),
            ResnetBlock(256, 256),
            nn.Conv2d(256, 512, 3, 2, 1),
            ResnetBlock(512, 512),
        ])

    def forward(self, x):

        logits = x
        block_outputs = []

        for block in self.network:
            logits = block(logits)
            if isinstance(block, ResnetBlock):
                block_outputs.append(logits)

        return logits, block_outputs


class Decoder(nn.Module):

    def __init__(self, out_channels: int):
        super(Decoder, self).__init__()

        self.network = nn.ModuleList([
            ResnetBlock(512, 512),
            nn.ConvTranspose2d(512, 256, 2, 2),
            ResnetBlock(256, 256),
            nn.ConvTranspose2d(256, 128, 2, 2),
            ResnetBlock(128, 128),
            nn.ConvTranspose2d(128, 64, 2, 2),
            ResnetBlock(64, out_channels),
        ])

    def forward(self, x, block_outputs):

        logits = x

        for block in self.network:
            if isinstance(block, ResnetBlock) and len(block_outputs):
                logits = block(logits, block_output=block_outputs.pop())
            else:
                logits = block(logits)

        return logits


class Unet(nn.Module):

    def __init__(self,in_channels: int, out_channels: int):
        super(Unet, self).__init__()

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        downsampled, block_outputs = self.encoder(x)
        upsampled = self.decoder(downsampled, block_outputs)
        probs = self.sigmoid(upsampled)

        return upsampled, probs
