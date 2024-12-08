from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm


class ScaleDiscriminator(nn.Module):
    """
    As described in MelGAN
    https://arxiv.org/pdf/1910.06711v3
    """

    def __init__(self, param_norm=weight_norm):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs += [
            param_norm(nn.Conv1d(1, 128, 15, stride=1, padding=7)),
            param_norm(nn.Conv1d(128, 128, 41, stride=2, padding=20, groups=4)),
            param_norm(nn.Conv1d(128, 256, 41, stride=2, padding=20, groups=16)),
            param_norm(nn.Conv1d(256, 512, 41, stride=4, padding=20, groups=16)),
            param_norm(nn.Conv1d(512, 1024, 41, stride=4, padding=20, groups=16)),
            param_norm(nn.Conv1d(1024, 1024, 41, stride=1, padding=20, groups=16)),
            param_norm(nn.Conv1d(1024, 1024, 5, stride=1, padding=2)),
        ]

        self.act = nn.LeakyReLU(0.1)
        self.classificator = param_norm(nn.Conv1d(1024, 1, 3, padding=1))
        self.flatten = nn.Flatten(1, -1)

    def forward(self, x):
        """
        Args:
            x (Tensor): input audio (B, 1, T).
        Returns:
            x (Tensor): real/fake logits (B).
            features (list[Tensor]): intermediate audio representations.
        """
        features = []
        for conv in self.convs:
            x = conv(x)
            x = self.act(x)
            features.append(x)

        x = self.classificator(x)
        features.append(x)

        x = self.flatten(x)

        return x, features


class MSD(nn.Module):
    """Multi-Scale Discriminator"""

    def __init__(self, num_scales=2):
        super().__init__()

        self.scale_discriminators = nn.ModuleList()
        self.scale_discriminators.append(ScaleDiscriminator(param_norm=spectral_norm))
        for _ in range(num_scales):
            self.scale_discriminators.append(ScaleDiscriminator())

        self.avg_pool = nn.AvgPool1d(4, 2, padding=2)

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

        for i, discr in enumerate(self.scale_discriminators):
            if i != 0:
                audio = self.avg_pool(audio)
                audio_fake = self.avg_pool(audio_fake)

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
