from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class TRANSFORMERModule(LightningModule):
    """LightningModule for Transformer."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        """Initialize a TRANSFORMERModule.

        Parameters
        ----------
        net
            The Transformer model.
        optimizer
            The optimizer.
        scheduler
            The learning rate scheduler.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        Parameters
        ----------
        x
            A tensor.

        Returns
        -------
        torch.Tensor
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""

        self.val_loss.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch
            A batch of data.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        x, y = batch
        x = x.permute(1, 0)
        y = y.permute(1, 0)
        # log.info(f"Input shape: {x.shape} | Labels shape: {y.shape}")
        logits = self.forward(x)
        # log.info(f"Logits shape: {logits.shape} | Labels shape: {y.shape}")
        loss = F.cross_entropy(logits.reshape(-1, 32000), y.reshape(-1))
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Parameters
        ----------
        batch
            A batch of data.
        batch_idx
            The index of the batch.

        Returns
        -------
        torch.Tensor
        """
        loss, preds, targets = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Parameters
        ----------
        batch
            A batch of data.
        batch_idx
            The index of the batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = TRANSFORMERModule(None, None, None)
