import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class CustomDirDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        data_dir,
        part="custom",
        target_sr=16000,
        shuffle_index=True,
        limit=None,
    ):
        """
        Args:
            data_dir (str): path to custom dataset.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self.target_sr = target_sr
        assert data_dir is not None, "You should provide path to dir"
        self.transcriptions_dir = Path(data_dir).absolute().resolve() / "transcriptions"

        self._index = self._get_index()

        if shuffle_index:
            random.seed(42)
            random.shuffle(self._index)

        if limit is not None:
            self._index = self._index[:limit]

    def _get_index(self):
        index = []

        for text_path in self.transcriptions_dir.iterdir():
            item = dict()
            item["utt_id"] = text_path.stem

            with open(text_path) as f:
                item["text"] = f.read()

            index.append(item)

        return index

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        item = self._index[ind]

        return {"text": item["text"], "utt_id": item["utt_id"], "sample_rate": self.target_sr}

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)


# ds = CustomDirDataset(data_dir="data/NameOfTheDirectoryWithUtterances", melspec_params=melspec_params)
# item = ds[0]
# print("index[0]", ds._index[0])
# print(item["text"])
# print(item["utt_id"])
# print(item["melspec"].shape)
