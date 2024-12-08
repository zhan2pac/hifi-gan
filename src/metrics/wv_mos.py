import torch
import warnings

warnings.filterwarnings("ignore")

from src.metrics.base_metric import BaseMetric
from src.metrics.wv_mos_model import Wav2Vec2MOS
from src.utils.io_utils import ROOT_PATH


class WVMOS(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """
        device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        path = self.load_weights()
        self.metric = Wav2Vec2MOS(path=path, freeze=True, cuda=True).to(device)

    def load_weights(self):
        import os
        import urllib.request

        path = ROOT_PATH / "data/wv_mos/wv_mos.ckpt"

        if not os.path.exists(path):
            print("Downloading the checkpoint for WV-MOS")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve("https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1", path)
            print("Weights downloaded in: {} Size: {}".format(path, os.path.getsize(path)))
        return path

    def __call__(self, audio_fake: torch.Tensor, **batch):
        """
        Args:
            audio_fake (Tensor): audio (B, 1, T).
        Returns:
            metric (float): calculated metric.
        """
        audio_fake = audio_fake.squeeze(1)
        return self.metric.calculate_batch(audio_fake)
