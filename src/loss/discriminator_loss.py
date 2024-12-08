import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def discriminator_loss(self, logits_real, logits_fake):
        total_loss = 0
        for real_logit, fake_logit in zip(logits_real, logits_fake):
            total_loss += torch.mean((1.0 - real_logit) ** 2) + torch.mean((fake_logit) ** 2)

        return total_loss

    def forward(self, mpd_output, msd_output, **batch):
        """
        Args:
            mpd_output (dict): logits and feature maps for real/fake audio from MPD.
            msd_output (dict): logits and feature maps for real/fake audio from MSD.
        Returns:
            output (dict): contains losses on real/fake audio and total loss.
        """
        mpd_loss = self.discriminator_loss(mpd_output["logits_real"], mpd_output["logits_fake"])
        msd_loss = self.discriminator_loss(msd_output["logits_real"], msd_output["logits_fake"])

        return {"d_loss": mpd_loss + msd_loss}
