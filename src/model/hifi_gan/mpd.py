import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, kernel_size, stride, param_norm=weight_norm):
        super().__init__()
        self.period = period
        pad = (kernel_size - 1) // 2

        self.convs = nn.ModuleList()
        self.convs += [
            param_norm(nn.Conv2d(1, 32, (kernel_size, 1), stride=(stride, 1), padding=(pad, 0))),
            param_norm(nn.Conv2d(32, 128, (kernel_size, 1), stride=(stride, 1), padding=(pad, 0))),
            param_norm(nn.Conv2d(128, 512, (kernel_size, 1), stride=(stride, 1), padding=(pad, 0))),
            param_norm(nn.Conv2d(512, 1024, (kernel_size, 1), stride=(stride, 1), padding=(pad, 0))),
            param_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), stride=(1, 1), padding=(pad, 0))),
        ]

        self.act = nn.LeakyReLU(0.1)
        self.classificator = param_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.flatten = nn.Flatten(1, -1)

    def pad_reshape(self, x):
        """
        Args:
            x (Tensor): input 1d audio (B, 1, T).
        Returns:
            x (Tensor): cutted into periods 2d audio (B, 1, T/P, P).
        """
        B, C, T = x.shape
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
            T = T + pad_len
        x = x.view(B, C, T // self.period, self.period)

        return x

    def forward(self, x):
        """
        Args:
            x (Tensor): input audio (B, 1, T).
        Returns:
            x (Tensor): real/fake logits (B).
            features (list[Tensor]): intermediate audio representations.
        """
        features = []
        x = self.pad_reshape(x)

        for conv in self.convs:
            x = conv(x)
            x = self.act(x)
            features.append(x)

        x = self.classificator(x)
        features.append(x)
        x = self.flatten(x)

        return x, features


class MPD(nn.Module):
    """Multi-Period Discriminator"""

    def __init__(self, periods=(2, 3, 5, 7, 11), kernel_size=5, stride=3):
        super().__init__()

        self.period_discriminators = nn.ModuleList()
        for p in periods:
            self.period_discriminators.append(PeriodDiscriminator(p, kernel_size, stride))

    def forward(self, audio, audio_fake):
        """
        Args:
            audio (Tensor): real audio.
            audio_fake (Tensor): generated audio.
        Returns:
            logits_real (list[Tensor]): logits of real audio.
            logits_fake (list[Tensor]): logits of generated audio.
            features_real (list[list[Tensor]]): features of real audio from intermediate layers of a discriminator.
            features_fake (list[list[Tensor]]): features of generated audio from intermediate layers of a discriminator.
        """
        logits_real = []
        logits_fake = []
        features_real = []
        features_fake = []

        for discr in self.period_discriminators:
            l_real, f_real = discr(audio)
            l_fake, f_fake = discr(audio_fake)

            logits_real.append(l_real)
            logits_fake.append(l_fake)
            features_real.append(f_real)
            features_fake.append(f_fake)

        return {
            "logits_real": logits_real,
            "logits_fake": logits_fake,
            "features_real": features_real,
            "features_fake": features_fake,
        }
