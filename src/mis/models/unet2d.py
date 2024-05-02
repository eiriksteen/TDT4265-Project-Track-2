import torch
import torch.nn.functional as F
import torch.nn as nn

class Swish(nn.Module):

    def forward(self, x):
        return x * F.sigmoid(x)

class ResnetBlock(nn.Module):

    def __init__(
            self, 
            in_channels, 
            out_channels,
            skip_conn = "concat"
            ):
        super(ResnetBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_conn = skip_conn

        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(32, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        self.adapt_shape = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=1) if in_channels != out_channels else nn.Identity()
        

    def forward(self, x, block_output=None):

        if block_output is not None:
            if self.skip_conn == "concat":
                x = torch.cat((x, block_output), dim=1)
            else:
                x = x + block_output

        logits = self.block(x)

        return logits + self.adapt_shape(x)


class Encoder(nn.Module):

    def __init__(self, in_channels: int, non_local=False):
        super(Encoder, self).__init__()

        self.network = nn.ModuleList([
            nn.Conv2d(in_channels, 64, 1, 1),
            ResnetBlock(64, 64),
            nn.Conv2d(64, 128, 3, 2, 1),
            ResnetBlock(128, 128),
            nn.Conv2d(128, 256, 3, 2, 1),
            NonLocalBlock(256) if non_local else nn.Identity(),
            ResnetBlock(256, 256),
            nn.Conv2d(256, 512, 3, 2, 1),
            NonLocalBlock(512) if non_local else nn.Identity(),
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

    def __init__(
            self, 
            out_channels: int, 
            skip_conn = "concat",
            non_local=False):
        super(Decoder, self).__init__()

        m = 2 if skip_conn == "concat" else 1
        self.network = nn.ModuleList([
            ResnetBlock(512*m, 512, skip_conn),
            NonLocalBlock(512) if non_local else nn.Identity(),
            nn.ConvTranspose2d(512, 256, 2, 2),
            ResnetBlock(256*m, 256, skip_conn),
            NonLocalBlock(256) if non_local else nn.Identity(),
            nn.ConvTranspose2d(256, 128, 2, 2),
            ResnetBlock(128*m, 128, skip_conn),
            nn.ConvTranspose2d(128, 64, 2, 2),
            ResnetBlock(64*m, 64, skip_conn),
            nn.Conv2d(64, out_channels, 1, 1),
        ])

    def forward(self, x, block_outputs):

        logits = x

        for block in self.network:
            if isinstance(block, ResnetBlock) and len(block_outputs):
                logits = block(logits, block_output=block_outputs.pop())
            else:
                logits = block(logits)

        return logits
    
class NonLocalBlock(nn.Module):

    # From https://github.com/dome272/VQGAN-pytorch/blob/main/helper.py

    def __init__(self, channels) -> None:
        super(NonLocalBlock, self).__init__()

        self.in_channels = channels

        self.gn = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A


class UNet2D(nn.Module):

    def __init__(
            self,
            in_channels: int, 
            out_channels: int,
            skip_conn: str = "concat"
            ):
        super(UNet2D, self).__init__()

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels, skip_conn=skip_conn)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        downsampled, block_outputs = self.encoder(x)
        upsampled = self.decoder(downsampled, block_outputs)
        probs = self.sigmoid(upsampled)

        return upsampled, probs
    
class UNet2DNonLocal(nn.Module):

    def __init__(
            self,
            in_channels: int, 
            out_channels: int,
            skip_conn: str = "concat"
            ):
        super(UNet2DNonLocal, self).__init__()

        self.encoder = Encoder(in_channels, non_local=True)
        self.decoder = Decoder(out_channels, non_local=True, skip_conn=skip_conn)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        downsampled, block_outputs = self.encoder(x)
        upsampled = self.decoder(downsampled, block_outputs)
        probs = self.sigmoid(upsampled)

        return upsampled, probs
