import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram


class GeneratorLoss(nn.Module):
    def __init__(self, melspec_params, lambda_feature=2.0, lambda_mel=45.0):
        super().__init__()
        self.lambda_feature = lambda_feature
        self.lambda_mel = lambda_mel

        if "f_max" in melspec_params:
            del melspec_params["f_max"]
        self.get_melspec = MelSpectrogram(**melspec_params)
        self.l1_loss = nn.L1Loss()

    def generator_loss(self, logits_fake):
        loss = 0
        for fake_logit in logits_fake:
            loss += torch.mean((fake_logit - 1) ** 2)

        return loss

    def feature_loss(self, features_real, features_fake):
        loss = 0
        for discr_features_real, discr_features_fake in zip(features_real, features_fake):
            for layer_feature_real, layer_feature_fake in zip(discr_features_real, discr_features_fake):
                loss += torch.mean(torch.abs(layer_feature_real - layer_feature_fake))
        return loss

    def forward(self, audio_fake, melspec_real, mpd_output, msd_output, **batch):
        """
        Args:
            audio_fake (Tensor): generated audio (B, 1, T).
            melspec_real (Tensor): mel-spectrogram of target audio (B, C, L).
            mpd_output (dict): logits and feature maps for real/fake audio from MPD.
            msd_output (dict): logits and feature maps for real/fake audio from MSD.
        Returns:
            output (dict): contains total generator loss and its indivisual parts.
        """
        adv_loss = 0
        adv_loss += self.generator_loss(mpd_output["logits_fake"])
        adv_loss += self.generator_loss(msd_output["logits_fake"])

        feature_loss = 0
        feature_loss += self.feature_loss(mpd_output["features_real"], mpd_output["features_fake"])
        feature_loss += self.feature_loss(msd_output["features_real"], msd_output["features_fake"])

        melspec_fake = self.get_melspec(audio_fake.squeeze(1))
        mel_loss = self.l1_loss(melspec_real, melspec_fake)

        total_loss = adv_loss + feature_loss * self.lambda_feature + mel_loss * self.lambda_mel

        return {
            "g_loss": total_loss,
            "g_adv_loss": adv_loss,
            "g_mel_loss": mel_loss,
            "g_feature_loss": feature_loss,
        }
