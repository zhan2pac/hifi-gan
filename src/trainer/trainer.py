import random

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer["G"].zero_grad()
            self.optimizer["D"].zero_grad()

        # ================ Discriminators step ================
        g_output = self.model.Generator(**batch)
        audio_fake = g_output["audio_fake"].detach()
        batch.update({"audio_fake": audio_fake})

        d_output = self.model.Discriminator(**batch)
        batch.update(d_output)

        d_losses = self.criterion["D"](**batch)
        batch.update(d_losses)

        if self.is_train:
            batch["d_loss"].backward()
            self._clip_grad_norm()
            self.optimizer["D"].step()
            if self.lr_scheduler["D"] is not None:
                self.lr_scheduler["D"].step()

        # ================ Generator step ================
        if self.is_train:
            self.optimizer["G"].zero_grad()

        batch.update(g_output)
        d_output = self.model.Discriminator(**batch)
        batch.update(d_output)

        g_losses = self.criterion["G"](**batch)
        batch.update(g_losses)

        if self.is_train:
            batch["g_loss"].backward()
            self._clip_grad_norm()
            self.optimizer["G"].step()
            if self.lr_scheduler["G"] is not None:
                self.lr_scheduler["G"].step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_audio(**batch)
        else:
            self.log_audio(**batch)

    def log_audio(self, audio, audio_fake, sample_rate, **batch):
        idx = random.randint(0, len(audio) - 1)
        gt_audio = audio[idx].detach().cpu()
        fake_audio = audio_fake[idx].detach().cpu()

        self.writer.add_audio("ground_truth/audio", gt_audio, sample_rate=sample_rate[idx])
        self.writer.add_audio("generated/audio", fake_audio, sample_rate=sample_rate[idx])
