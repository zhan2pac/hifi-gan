import torch
import torchaudio
from tqdm.auto import tqdm

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.io_utils import ROOT_PATH


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
        writer=None,
        use_tts=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
            writer (WandBWriter): optionally log predictions to wandb server
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        self.writer = writer
        self.use_tts = use_tts

        # path definition
        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = MetricTracker(writer=None)

        if not skip_model_load:
            # init model
            pretrained_path = ROOT_PATH / config.inferencer.from_pretrained
            self._from_pretrained(pretrained_path)

        if self.use_tts:
            self.initialize_tacotron2()

    def initialize_tacotron2(self):
        print("Downloading Tacotron2...")
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        self.processor = bundle.get_text_processor()
        self.tacotron2 = bundle.get_tacotron2().to(self.device)

    def generate_melspec(self, text):
        with torch.inference_mode():
            processed, lengths = self.processor(text)
            processed = processed.to(self.device)
            lengths = lengths.to(self.device)
            spec, _, _ = self.tacotron2.infer(processed, lengths)

        return spec

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        if self.use_tts:
            melspec = self.generate_melspec(batch["text"])  # [1, C, L]
            batch["melspec"] = melspec

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        g_output = self.model.Generator(**batch)
        batch.update(g_output)

        if self.metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        batch_size = batch["melspec"].shape[0]

        for idx in range(batch_size):
            if part != "custom":
                gt_audio = batch["audio"][idx].detach().cpu().unsqueeze(0)  # [1, L]
            fake_audio = batch["audio_fake"][idx].detach().cpu().unsqueeze(0)

            sr = batch["sample_rate"][idx]
            utt_id = batch["utt_id"][idx]

            if self.save_path is not None:
                self.save_audio(self.save_path / part / "fake_audio", utt_id, fake_audio, sr)
                if part != "custom":
                    self.save_audio(self.save_path / part / "gt_audio", utt_id, gt_audio, sr)

            if self.writer is not None:
                self.writer.add_audio(f"generated/audio_{utt_id}", fake_audio, sample_rate=sr)
                if part != "custom":
                    self.writer.add_audio(f"ground_truth/audio_{utt_id}", gt_audio, sample_rate=sr)

                melspec = batch["melspec"][idx].detach().cpu()  # [C, L]
                self.writer.add_image(f"spectrogram/spec_{utt_id}", plot_spectrogram(melspec))

        return batch

    def save_audio(self, path, utt_id, audio_tensor, sr):
        path.mkdir(exist_ok=True, parents=True)
        torchaudio.save(path / (utt_id + ".wav"), audio_tensor, sr)

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
