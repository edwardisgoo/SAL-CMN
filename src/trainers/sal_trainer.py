import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
import matplotlib.pyplot as plt

from src.evaluator.evaluator import EERMetric, F1Metric, EERMetricWithLabel, F1MetricWithLabel
from src.models.criterion.basic_loss import MaskedMultiClassCrossEntropyLoss
from src.utils import mixup
from src.trainers.label_generators import SPLLabelGenerator, TransitionLabelGenerator

class SALTrainer(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler = None,
            v2: int = 1,
            mixup: bool = False,
            mixup2: bool = False,
            mixup_ratio: float = 0.5,
            compile: bool = False,
            vis: bool = False,
            vis_top_k_hard: int = 10,
            vis_top_k_easy: int = 5,
    ) -> None:
        """Initialize the SAL trainer (SPL + CSM).

        :param net: The model to train.
        :param criterion: The loss function to use for training.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param mixup: Whether to apply mixup data augmentation.
        :param mixup_ratio: The ratio of samples in a batch to apply mixup to (0.0 to 1.0).
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])
        self.v2 = v2
        self.mixup = mixup
        self.mixup2 = mixup2
        self.mixup_ratio = mixup_ratio
        self.vis = vis
        self.vis_top_k_hard = vis_top_k_hard
        self.vis_top_k_easy = vis_top_k_easy
        # load model, criterion, optimizer, and scheduler
        self.net = net
        self.criterion = MaskedMultiClassCrossEntropyLoss(num_classes=8)
        self.criterion2 = MaskedMultiClassCrossEntropyLoss(num_classes=2)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking test metrics
        self.val_eer = EERMetric()
        self.val_eer_best = MinMetric()
        self.test_eer = EERMetric()
        self.val_acc = F1Metric()
        self.test_acc = F1Metric()

        # Poisoning prevention flags/state
        self._skip_optimizer_step: bool = False
        self._last_bad_utt_ids: list = []
        self._last_good_weight_layer: torch.Tensor | None = None

    def _mixup_batch(
            self,
            batch: Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]:
        return mixup.mixup_batch(batch, mixup_ratio=self.mixup_ratio, cut_min=0.2, cut_max=0.8)

    def _get_label_mask(
            self,
            labels: torch.Tensor,
            label_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized mask: True for valid positions, False for padding."""
        B, L = labels.shape[:2]
        ar = torch.arange(L, device=labels.device).unsqueeze(0)
        mask = ar < label_lengths.view(-1, 1)
        return mask

    def _get_pred_label(
            self,
            preds: torch.Tensor,
            labels: torch.Tensor,
            label_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the predicted labels and their corresponding ground truth labels.

        :param preds: A tensor of model predictions.
        :param labels: A tensor of ground truth labels.
        :param label_lengths: A tensor of label lengths.
        :return: A tuple containing a list of predicted labels and a list of
            ground truth labels.
        """
        pred_list = []
        label_list = []
        for i, length in enumerate(label_lengths):
            pred_list.append(preds[i, :length])
            label_list.append(labels[i, :length])

        preds_flat = torch.cat(pred_list, dim=0)
        labels_flat = torch.cat(label_list, dim=0).float()
        return preds_flat, labels_flat

    @property
    def exp_dir(self) -> str:
        if hasattr(self.trainer, "ckpt_path") and self.trainer.ckpt_path:
            # Use the directory of the checkpoint file as the root
            return str(Path(self.trainer.ckpt_path).parent.parent)
        else:
            # Fallback to logger's log_dir
            log_dir = self.logger.log_dir
            return str.join("/", log_dir.split("/")[:-2])

    def _log_warn(self, msg: str) -> None:
        """Print and also append a warning line to exp_dir/nan_warnings.log."""
        try:
            print(msg, flush=True)
        except Exception:
            pass
        try:
            exp_dir = self.exp_dir
            os.makedirs(exp_dir, exist_ok=True)
            with open(os.path.join(exp_dir, "nan_warnings.log"), "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_eer.reset()
        self.val_eer_best.reset()
        self.val_acc.reset()

    def training_step(
            self, batch: Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the
        training set.

        :param batch: A batch of data (a tuple) containing the input tensor
        of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # Apply mixup if enabled
        if self.mixup:
            batch = self._mixup_batch(batch)
        if self.mixup2:
            batch = self._mixup_batch(batch)
            batch = self._mixup_batch(batch)
        utt_ids, inputs, labels, label_lengths = batch

        # Guard 1: check inputs for NaN/Inf prior to forward
        if not torch.isfinite(inputs).all():
            flat = inputs.view(inputs.size(0), -1)
            bad = ~torch.isfinite(flat).all(dim=1)
            bad_ids = [utt_ids[i] for i in torch.nonzero(bad, as_tuple=False).squeeze(-1).tolist()]
            self._log_warn(f"WARN[NAN] inputs epoch={getattr(self, 'current_epoch', -1)} batch={batch_idx} step={getattr(self, 'global_step', -1)} ids={bad_ids[:16]}")
            # Drop bad samples from the batch
            keep = ~bad
            inputs = inputs[keep]
            labels = labels[keep]
            label_lengths = label_lengths[keep]
            utt_ids = [u for k, u in zip(keep.tolist(), utt_ids) if k]
            if inputs.numel() == 0 or inputs.size(0) == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

        preds1, preds2 = self.forward(inputs)
        # Snapshot last good weight_layer to allow recovery if poisoned
        try:
            wl = getattr(self.net, "weight_layer", None)
            if wl is not None and torch.isfinite(wl).all():
                self._last_good_weight_layer = wl.detach().clone()
        except Exception:
            pass

        # Guard 2: check preds for NaN/Inf
        any_bad = False
        for p, tag in ((preds1, 'preds1'), (preds2, 'preds2')):
            if not torch.isfinite(p).all():
                flat = p.view(p.size(0), -1)
                bad = ~torch.isfinite(flat).all(dim=1)
                bad_ids = [utt_ids[i] for i in torch.nonzero(bad, as_tuple=False).squeeze(-1).tolist()]
                self._log_warn(f"WARN[NAN] {tag} epoch={getattr(self, 'current_epoch', -1)} batch={batch_idx} step={getattr(self, 'global_step', -1)} ids={bad_ids[:16]}")
                # Drop bad samples
                keep = ~bad
                preds1 = preds1[keep]
                preds2 = preds2[keep]
                labels = labels[keep]
                label_lengths = label_lengths[keep]
                utt_ids = [u for k, u in zip(keep.tolist(), utt_ids) if k]
                any_bad = True
                if preds1.size(0) == 0:
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        mask = self._get_label_mask(labels, label_lengths)
        if mask.sum().item() == 0:
            self._log_warn(f"WARN[EMPTY_MASK] epoch={getattr(self, 'current_epoch', -1)} batch={batch_idx} step={getattr(self, 'global_step', -1)} ids={utt_ids[:16]}")
        
        # Generate boundary labels for batch
        labels_info, lengths_info = SPLLabelGenerator._seg2bd_label_new(labels)
        target_batch = SPLLabelGenerator.seg2bd_label_new(labels_info, lengths_info)
        
        # Convert batch targets to tensor format
        if isinstance(target_batch, list):
            target_list = []
            for i, target_seq in enumerate(target_batch):
                target_tensor = torch.tensor(target_seq, device=preds1.device).squeeze()
                target_list.append(target_tensor)
            target = torch.cat(target_list, dim=0).type(torch.long)
        else:
            target = torch.tensor(target_batch, device=preds1.device).view(-1).type(torch.long)
        
        loss1 = self.criterion(preds1.transpose(1, 2),
                             target.reshape(labels.shape[0], -1).to(torch.long), mask=mask)
        loss2 = self.criterion2(preds2,
                             labels.to(torch.long), mask=mask)
        loss = loss1 + self.v2 * loss2

        # Guard 3: ensure finite loss
        if not torch.isfinite(loss):
            self._log_warn(f"WARN[NAN_LOSS] epoch={getattr(self, 'current_epoch', -1)} batch={batch_idx} step={getattr(self, 'global_step', -1)} ids={utt_ids[:16]}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, on_epoch=True,
                 sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.train_loss.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, list],
                        batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the
        validation set.

        :param batch: A batch of data (a tuple) containing the input tensor
        of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        utt_ids, inputs, labels, label_lengths = batch
        preds1, preds2 = self.forward(inputs)
        mask = self._get_label_mask(labels, label_lengths)
        
        # Generate boundary labels for batch
        labels_info, lengths_info = SPLLabelGenerator._seg2bd_label_new(labels)
        target_batch = SPLLabelGenerator.seg2bd_label_new(labels_info, lengths_info)
        
        # Convert batch targets to tensor format
        if isinstance(target_batch, list):
            # Handle batch data
            target_list = []
            for i, target_seq in enumerate(target_batch):
                target_tensor = torch.tensor(target_seq, device=preds1.device).squeeze()
                target_list.append(target_tensor)
            target = torch.cat(target_list, dim=0).type(torch.long)
        else:
            # Handle single sequence
            target = torch.tensor(target_batch, device=preds1.device).view(-1).type(torch.long)
        loss1 = self.criterion(preds1.transpose(1, 2),
                              target.reshape(labels.shape[0], -1).to(torch.long), mask=mask)
        loss2 = self.criterion2(preds2,
                              labels.to(torch.long), mask=mask)
        loss = loss1 + self.v2 * loss2
        # update and log metrics
        preds_flat, labels_flat = self._get_pred_label(preds2, labels,
                                                       label_lengths)
        self.val_loss(loss)
        self.val_eer.update(preds_flat[:, 1] - preds_flat[:, 0], labels_flat)
        self.val_acc.update(preds_flat, labels_flat)
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        eer, thresh = self.val_eer.compute()  # get current val eer
        self.val_eer_best(eer)  # update best so far val eer
        acc, f1 = self.val_acc.compute()
        self.log("val/eer", eer, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        self.log("val/eer_best", self.val_eer_best.compute(),
                 prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        self.val_loss.reset()
        self.val_eer.reset()
        self.val_acc.reset()

    def on_test_start(self) -> None:
        """Lightning hook that is called when testing begins."""
        self.test_loss.reset()
        self.test_eer.reset()
        self.test_acc.reset()
        if self.vis:
            self._vis_items = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, list],
                  batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor
        of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        utt_ids, inputs, labels, label_lengths = batch
        preds1, preds2 = self.forward(inputs)
        mask = self._get_label_mask(labels, label_lengths)
        
        # Generate boundary labels for batch
        labels_info, lengths_info = SPLLabelGenerator._seg2bd_label_new(labels)
        target_batch = SPLLabelGenerator.seg2bd_label_new(labels_info, lengths_info)
        
        # Convert batch targets to tensor format
        if isinstance(target_batch, list):
            # Handle batch data
            target_list = []
            for i, target_seq in enumerate(target_batch):
                target_tensor = torch.tensor(target_seq, device=preds1.device).reshape(-1)
                target_list.append(target_tensor)
            target = torch.cat(target_list, dim=0).type(torch.long)
        else:
            # Handle single sequence
            target = torch.tensor(target_batch, device=preds1.device).view(-1).type(torch.long)
        
        loss1 = self.criterion(preds1.transpose(1, 2),
                              target.reshape(labels.shape[0], -1).to(torch.long), mask=mask)
        loss2 = self.criterion2(preds2,
                              labels.to(torch.long), mask=mask)
        loss = loss1 + self.v2 * loss2
        # update and log metrics
        preds_flat, labels_flat = self._get_pred_label(preds2, labels,
                                                       label_lengths)
        preds_flat8, labels_flat8 = self._get_pred_label(preds1, target.reshape(labels.shape[0], -1),
                                                       label_lengths)
        self.test_loss(loss)
        
        self.test_eer.update(preds_flat[:, 1] - preds_flat[:, 0], labels_flat)
        self.test_acc.update(preds_flat, labels_flat)
        self.log("test/loss", self.test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        if self.vis:
            with torch.no_grad():
                B = labels.shape[0]
                for i in range(B):
                    L = int(label_lengths[i].item())
                    if L <= 0:
                        continue
                    scores = (preds2[i, :L, 1] - preds2[i, :L, 0]).detach().cpu()
                    labs = labels[i, :L].to(torch.long).detach().cpu()
                    utt_id = utt_ids[i] if isinstance(utt_ids, (list, tuple)) else str(i)
                    self._vis_items.append({"utt_id": utt_id, "scores": scores, "labels": labs})

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        eer, thresh_eer = self.test_eer.compute()
        acc, f1 = self.test_acc.compute()
        self.log("test/eer", eer, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        result_file = os.path.join(self.exp_dir, "result.txt")
        with open(result_file, "a") as f:
            f.write(f"EER={eer:.4f} ACC={acc:.4f} F1={f1:.4f}\n")

        if self.vis and hasattr(self, "_vis_items") and len(self._vis_items) > 0:
            vis_dir = os.path.join(self.exp_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            # compute per-utt error rate
            errors = []
            for item in self._vis_items:
                scores = item["scores"].numpy()
                labs = item["labels"].numpy().astype(int)
                preds = (scores > float(thresh_eer)).astype(int)
                err = float(np.mean(np.array((preds != labs)))) if labs.size > 0 else 1.0
                errors.append(err)
            ranked = list(enumerate(errors))
            ranked.sort(key=lambda x: x[1], reverse=True)
            hard_idx = [i for i, _ in ranked[:self.vis_top_k_hard]]
            easy_idx = [i for i, _ in sorted(ranked[-self.vis_top_k_easy:], key=lambda x: x[1])]
            for tag, idx_list in (("hard", hard_idx), ("easy", easy_idx)):
                for pos, idx in enumerate(idx_list, start=1):
                    item = self._vis_items[idx]
                    scores = item["scores"].numpy()
                    labs = item["labels"].numpy().astype(int)
                    T = labs.shape[0]
                    xs = np.arange(T)
                    plt.figure(figsize=(12, 4))
                    plt.plot(xs, scores, color='tab:blue', linewidth=1.5, label='score (pos-neg)')
                    plt.axhline(float(thresh_eer), color='gray', linestyle='--', label=f'thresh={float(thresh_eer):.3f}')
                    ones = np.where(labs == 1)[0]
                    zeros = np.where(labs == 0)[0]
                    ax = plt.gca()
                    y_min, y_max = ax.get_ylim()
                    if ones.size > 0:
                        plt.scatter(ones, np.full_like(ones, y_max, dtype=float), s=8, c='tab:green', label='GT=1')
                    if zeros.size > 0:
                        plt.scatter(zeros, np.full_like(zeros, y_min, dtype=float), s=8, c='tab:red', label='GT=0')
                    plt.title(f"{tag.upper()} | utt={item['utt_id']} | err={errors[idx]:.3f}")
                    plt.xlabel("frame")
                    plt.ylabel("score / GT")
                    plt.legend(loc='best')
                    plt.tight_layout()
                    save_path = os.path.join(vis_dir, f"{tag}_{pos:02d}_{str(item['utt_id']).replace('/', '_')}.png")
                    plt.savefig(save_path, dpi=200)
                    plt.close()

        self.test_loss.reset()
        self.test_eer.reset()
        self.test_acc.reset()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train +
        validate), validate, test, or predict.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
                                                                                                                      
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizers and learning-rate schedulers to be used

        :return: A dict containing the configured optimizers and
        learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.net.parameters())

        if self.hparams.scheduler is not None:
            total_steps = self.trainer.estimated_stepping_batches
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
    
