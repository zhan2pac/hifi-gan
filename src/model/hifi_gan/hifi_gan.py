from torch import nn

from .generator import Generator
from .mpd import MPD
from .msd import MSD


class HiFiGAN(nn.Module):
    def __init__(self, generator_config, discriminator_config):
        super().__init__()
        self.Generator = Generator(**generator_config)
        self.Discriminator = Discriminator(**discriminator_config)


class Discriminator(nn.Module):
    def __init__(self, mpd_config, msd_config):
        super().__init__()
        self.multi_period_discriminator = MPD(**mpd_config)
        self.multi_scale_discriminator = MSD(**msd_config)

    def forward(self, audio, audio_fake, **batch):
        """
        Args:
            audio (Tensor): real audio (B, T).
            audio_fake (Tensor): generated audio (B, T).
        Returns:
            mpd_output (dict): logits and feature maps for real/fake audio from MPD.
            msd_output (dict): logits and feature maps for real/fake audio from MSD.
        """
        audio = audio.unsqueeze(1)
        audio_fake = audio_fake.unsqueeze(1)

        mpd_output = self.multi_period_discriminator(audio, audio_fake)
        msd_output = self.multi_scale_discriminator(audio, audio_fake)

        return {"mpd_output": mpd_output, "msd_output": msd_output}
