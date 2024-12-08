import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from .resblock import ResBlock


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class MRF(nn.Module):
    """Multi-Receptive Field Fusion Module"""

    def __init__(self, in_channels, kernel_sizes, dilation_sizes, resblock_type):
        super().__init__()
        self.num_blocks = len(kernel_sizes)

        self.conv_blocks = nn.ModuleList()
        for kernel_size, dilation_size in zip(kernel_sizes, dilation_sizes):
            self.conv_blocks.append(ResBlock(in_channels, kernel_size, dilation_size, resblock_type))

    def forward(self, x):
        out_sum = 0.0
        for resblock in self.conv_blocks:
            out_sum += resblock(x)

        return out_sum / self.num_blocks


class Generator(nn.Module):
    def __init__(
        self,
        n_mels,
        hidden_dim,
        upsample_rates,
        kernel_sizes_upsample,
        kernel_sizes_resblock,
        dilation_sizes_resblock,
        resblock_type="v1",
    ):
        super().__init__()
        self.feature_proj = weight_norm(nn.Conv1d(n_mels, hidden_dim, kernel_size=7, padding=3))
        self.act = nn.LeakyReLU(0.1)

        self.upsample_blocks = nn.ModuleList()
        self.mrf_blocks = nn.ModuleList()

        for i, (stride, kernel_size) in enumerate(zip(upsample_rates, kernel_sizes_upsample)):
            in_channels = hidden_dim // (2**i)
            out_channels = hidden_dim // (2 ** (i + 1))

            self.upsample_blocks.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(kernel_size - stride) // 2,
                    )
                )
            )
            self.mrf_blocks.append(MRF(out_channels, kernel_sizes_resblock, dilation_sizes_resblock, resblock_type))

        self.output_proj = weight_norm(nn.Conv1d(out_channels, 1, kernel_size=7, padding=3))
        self.tanh = nn.Tanh()

        self.upsample_blocks.apply(init_weights)
        self.output_proj.apply(init_weights)

    def forward(self, melspec, **batch):
        """
        Args:
            melspec (Tensor): real mel-spectrogram (B, C, L).
        Returns:
            audio_fake (Tensor): generated audio (B, T).
        """
        x = self.feature_proj(melspec)

        for upsample, mrf in zip(self.upsample_blocks, self.mrf_blocks):
            x = self.act(x)
            x = upsample(x)
            x = mrf(x)

        x = self.act(x)
        x = self.output_proj(x)
        x = self.tanh(x).squeeze(1)

        x = self.peak_normalize(x)

        return {"audio_fake": x}

    def peak_normalize(self, tensor):
        # https://discuss.pytorch.org/t/how-to-normalize-audio-data-in-pytorch/187709/2
        tensor = tensor - torch.mean(tensor)
        return tensor / torch.max(torch.abs(tensor))
