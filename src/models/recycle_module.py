from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.FastRCNN import FastRCNN

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class RecycleLitModule(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # # loss function
        # self.criterion = torch.nn.CrossEntropyLoss()

        self.loss_hist = Averager()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, images, targets):
        return self.net.forward(images, targets)

    def training_step(self, batch: Any, batch_idx: int):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.forward(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        # log train metrics
        self.log("loss", losses, on_step=False, on_epoch=True, prog_bar=False)
        self.log("log", loss_dict, on_step=False, on_epoch=True, prog_bar=True)
    
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.forward(images, targets)
    
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # log val metrics
        #acc = self.val_acc(preds, targets)
        #self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {}

    def validation_epoch_end(self, outputs: List[Any]):
        #self.coco_evaluator.accumulate()
        #self.coco_evaluator.summarize()
        ##coco main metric
        #metric = self.coco_evaluator.coco_eval['bbox'].stats[0]
        metric = 0
        self.log("main_score", metric, on_step=False, on_epoch=True, prog_bar=False)
        return {}

    def test_step(self, batch: Any, batch_idx: int):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.forward(images, targets)

        # log test metrics
        #acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        #self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()
        self.loss_hist.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.SGD(
            params=self.net.parameters(), momentum=self.hparams.momentum, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
