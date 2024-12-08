import random
import torch

import torchaudio
import torch.nn.functional as F

from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset

from src.utils.io_utils import ROOT_PATH

MAX_WAV_VALUE = 32768.0


class MelDataset(Dataset):
    def __init__(
        self,
        part,
        melspec_params,
        data_dir=None,
        shuffle_index=True,
        segment_size=None,
    ):
        self.segment_size = segment_size
        self.target_sr = melspec_params["sample_rate"]

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "LJSpeech-1.1"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._index = self._get_index(part)

        random.seed(1234)
        if shuffle_index:
            random.shuffle(self._index)

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

        melspec = self.get_melspec(audio)
        target_melspec = self.get_target_melspec(audio)

        return {
            "melspec": melspec,
            "audio": audio,
            "melspec_real": target_melspec,
            "sample_rate": self.target_sr,
            "wav_path": wav_path,
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
        length = torch.max(torch.abs(audio), dim=1).values
        audio_norm = audio / length
        return audio_norm


# ds = MelDataset(part="train")
# item = ds[0]

# print("index[0]", ds._index[0])
# print(item["x_mel"].shape, item["y"].shape, item["target_mel"].shape)
# print("min", item["y"].min(), "max", item["y"].max())
