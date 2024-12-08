import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram

from src.utils.io_utils import ROOT_PATH

MAX_WAV_VALUE = 32768.0


class MelDataset(Dataset):
    def __init__(
        self,
        part,
        melspec_params,
        data_dir=None,
        segment_size=None,
        shuffle_index=True,
        limit=None,
    ):
        self.segment_size = segment_size
        self.target_sr = melspec_params["sample_rate"]

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "LJSpeech-1.1"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._index = self._get_index(part)

        if shuffle_index:
            random.seed(42)
            random.shuffle(self._index)

        if limit is not None:
            self._index = self._index[:limit]

        self.get_melspec = MelSpectrogram(**melspec_params)
        del melspec_params["f_max"]
        self.get_target_melspec = MelSpectrogram(**melspec_params)

    def _get_index(self, part):
        wavs_dir = self._data_dir / "wavs"

        part_file = self._data_dir / (part + ".txt")
        with open(part_file, "r") as f:
            index = []
            for line in f.read().split("\n"):
                if len(line) > 0:
                    index.append(wavs_dir / (line.split("|")[0] + ".wav"))
        return index

    def __getitem__(self, ind):
        wav_path = self._index[ind]

        audio = self.load_audio(wav_path)
        audio = audio / MAX_WAV_VALUE
        audio = self.normalize(audio) * 0.95

        audio = torch.FloatTensor(audio)
        if self.segment_size is not None:
            if audio.size(1) >= self.segment_size:
                start = random.randint(0, audio.size(1) - self.segment_size)
                audio = audio[:, start : start + self.segment_size]
            else:
                audio = F.pad(audio, (0, self.segment_size - audio.size(1)), value=0.0)

        melspec = self.get_melspec(audio).clamp_(min=1e-5).log_()
        target_melspec = self.get_target_melspec(audio).clamp_(min=1e-5).log_()

        return {
            "melspec": melspec,
            "audio": audio,
            "melspec_real": target_melspec,
            "sample_rate": self.target_sr,
            "wav_path": wav_path.name,
        }

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(str(path))

        if audio_tensor.size(0) != 1:
            audio_tensor = audio_tensor.mean(0, keepdim=True)

        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.target_sr)
        return audio_tensor

    def normalize(self, audio):
        length = torch.max(torch.abs(audio))
        audio_norm = audio / length
        return audio_norm


# from pathlib import Path

# ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent

# melspec_params = {
#     "sample_rate": 22050,
#     "n_fft": 1024,
#     "win_length": 1024,
#     "hop_length": 256,
#     "pad": 384,  # (n_fft - hop_length) // 2
#     "n_mels": 80,
#     "center": False,
#     "f_max": 8000,
# }

# ds = MelDataset(part="test", melspec_params=melspec_params)
# item = ds[0]
# print("index[0]", ds._index[0])
# print(item["wav_path"])
