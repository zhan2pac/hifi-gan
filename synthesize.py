import logging
import os
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchinfo import summary

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    project_config = OmegaConf.to_container(config)
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    writer = instantiate(config.writer, logger, project_config)

    device_ids = config.inferencer.device_ids
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(str, device_ids))
    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    summary(model)

    # get metrics
    metrics = instantiate(config.metrics)

    # save_path for model predictions
    save_path = ROOT_PATH / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
        writer=writer,
        use_tts=True,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
