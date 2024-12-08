import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}

    result_batch["audio"] = torch.cat([elem["audio"] for elem in dataset_items], dim=0)  # [B, T]
    result_batch["melspec"] = torch.cat([elem["melspec"] for elem in dataset_items], dim=0)  # [B, C, L]
    result_batch["melspec_real"] = torch.cat([elem["melspec_real"] for elem in dataset_items], dim=0)  # [B, C, L]

    result_batch["sample_rate"] = [elem["sample_rate"] for elem in dataset_items]
    result_batch["wav_path"] = [elem["wav_path"] for elem in dataset_items]

    return result_batch
