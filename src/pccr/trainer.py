from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch
from lightning import LightningModule

from src.loss import DiceScore
from src.pccr.config import PCCRConfig
from src.pccr.losses import RegistrationCriterion, SyntheticTargets
from src.pccr.model import PCCRModel
from src.utils import get_one_hot


class LiTPCCR(LightningModule):
    def __init__(self, args, config: PCCRConfig, experiment_logger=None):
        super().__init__()
        self.args = args
        if not hasattr(self.args, "phase"):
            self.args.phase = getattr(config, "phase", "real")
        if not hasattr(self.args, "lr"):
            self.args.lr = 1e-4
        if not hasattr(self.args, "max_epochs"):
            self.args.max_epochs = 1
        self.config = config
        self.model = PCCRModel(config)
        self.criterion = RegistrationCriterion(
            image_size=config.data_size,
            phase=args.phase,
            image_loss_weight=config.image_loss_weight,
            multiscale_similarity_factors=config.multiscale_similarity_factors,
            multiscale_similarity_weights=config.multiscale_similarity_weights,
            segmentation_supervision_weight=config.segmentation_supervision_weight,
            smoothness_weight=config.smoothness_weight,
            jacobian_weight=config.jacobian_weight,
            inverse_consistency_weight=config.inverse_consistency_weight,
            correspondence_weight=config.correspondence_weight,
            residual_velocity_weight=config.residual_velocity_weight,
            num_labels=config.num_labels,
        )
        self.experiment_logger = experiment_logger
        self.lr = args.lr
        self.test_outputs = []
        self._freeze_state = ""
        if self.args.phase == "real" and self.config.refinement_warmup_epochs > 0:
            self._freeze_encoder_and_canonical_heads()

    def _log_metrics(self, metrics: dict[str, float | torch.Tensor], step: int | None = None):
        if self.experiment_logger is None:
            return
        serialized = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in metrics.items()
        }
        self.experiment_logger.log_metrics(serialized, step=step)

    def log_aim_image(self, image: np.ndarray, name: str, step: int | None = None, context: dict | None = None) -> bool:
        if self.experiment_logger is None:
            return False

        run = None
        for candidate in (
            getattr(self.experiment_logger, "experiment", None),
            getattr(self.experiment_logger, "_run", None),
            getattr(self.experiment_logger, "run", None),
            self.experiment_logger,
        ):
            if candidate is not None and hasattr(candidate, "track"):
                run = candidate
                break
        if run is None:
            return False

        try:
            from aim import Image as AimImage
        except Exception:
            return False

        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        try:
            run.track(AimImage(image), name=name, step=step, context=context or {})
            return True
        except Exception:
            return False

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        return self.model(source, target)

    def _compute_loss(self, batch):
        source, target, source_label, target_label = batch[:4]
        outputs = self(source, target)
        synthetic_targets = None
        if self.args.phase == "synthetic":
            synthetic_targets = SyntheticTargets(
                canonical_source=batch[4],
                canonical_target=batch[5],
                valid_source=batch[6],
                valid_target=batch[7],
                matchability_source=batch[8],
                matchability_target=batch[9],
            )
        losses = self.criterion(
            outputs,
            source,
            target,
            source_label=source_label,
            target_label=target_label,
            synthetic_targets=synthetic_targets,
        )
        return outputs, losses, source_label, target_label

    def _set_requires_grad(self, module: nn.Module, enabled: bool):
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    def _freeze_encoder_and_canonical_heads(self):
        self._set_requires_grad(self.model.encoder, False)
        self._set_requires_grad(self.model.pointmap_head, False)
        self._set_requires_grad(self.model.decoder, True)
        self._freeze_state = "warmup"

    def _unfreeze_all_modules(self):
        self._set_requires_grad(self.model, True)
        self._freeze_state = "full"

    def _apply_training_stage(self):
        if self.args.phase != "real" or self.config.refinement_warmup_epochs <= 0:
            if self._freeze_state != "full":
                self._unfreeze_all_modules()
            return
        if self.current_epoch < self.config.refinement_warmup_epochs:
            if self._freeze_state != "warmup":
                self._freeze_encoder_and_canonical_heads()
                self.print(
                    f"[LiTPCCR] Warmup phase active through epoch "
                    f"{self.config.refinement_warmup_epochs - 1}: encoder and canonical heads frozen."
                )
        elif self._freeze_state != "full":
            self._unfreeze_all_modules()
            self.print("[LiTPCCR] Switched to end-to-end fine-tuning.")

    def on_fit_start(self):
        self._apply_training_stage()

    def on_train_epoch_start(self):
        self._apply_training_stage()

    def training_step(self, batch, batch_idx):
        _, losses, _, _ = self._compute_loss(batch)
        self.log("train_loss", losses["avg_loss"], prog_bar=True, on_epoch=True, on_step=True)
        self._log_metrics({f"train_{k}": v for k, v in losses.items()}, step=self.global_step)
        return losses["avg_loss"]

    def validation_step(self, batch, batch_idx):
        outputs, losses, source_label, target_label = self._compute_loss(batch)
        for name, value in losses.items():
            self.log(f"val_{name}", value, prog_bar=name == "avg_loss", on_step=False, on_epoch=True)

        if self.args.phase != "synthetic":
            score = self._dice_from_outputs(outputs, source_label, target_label)
            self.log("val_dice", score.mean(), prog_bar=True, on_step=False, on_epoch=True)
            self._log_metrics({"val_dice": score.mean()}, step=self.global_step)

        self._log_metrics({f"val_{k}": v for k, v in losses.items()}, step=self.global_step)
        return losses["avg_loss"]

    def test_step(self, batch, batch_idx):
        outputs, _, source_label, target_label = self._compute_loss(batch)
        score = self._dice_from_outputs(outputs, source_label, target_label).mean()
        self.test_outputs.append(score)
        self.log("test_dice", score, prog_bar=True, on_epoch=True, on_step=False)
        self._log_metrics({"test_dice": score}, step=self.global_step)
        return score

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return
        avg_score = torch.stack(self.test_outputs).mean()
        self.log("avg_test_dice", avg_score, prog_bar=True)
        self._log_metrics({"avg_test_dice": avg_score}, step=self.global_step)
        self.test_outputs.clear()

    def _dice_from_outputs(self, outputs, source_label, target_label):
        forward_disp = outputs["phi_s2t"]
        source_onehot = get_one_hot(source_label, self.config.num_labels)
        warped_channels = []
        for channel_id in range(self.config.num_labels):
            warped_channels.append(
                self.model.decoder.final_transformer(
                    source_onehot[:, channel_id : channel_id + 1].float(),
                    forward_disp.float(),
                )
            )
        warped_seg = torch.cat(warped_channels, dim=1)
        return DiceScore(warped_seg, target_label.long(), self.config.num_labels)

    def configure_optimizers(self):
        parameter_groups = {
            "encoder": {"params": [], "lr": self.lr * self.config.encoder_lr_scale},
            "pointmap": {"params": [], "lr": self.lr * self.config.canonical_head_lr_scale},
            "decoder": {"params": [], "lr": self.lr * self.config.coarse_decoder_lr_scale},
            "refinement": {"params": [], "lr": self.lr * self.config.residual_refinement_lr_scale},
            "other": {"params": [], "lr": self.lr},
        }

        for name, parameter in self.named_parameters():
            if name.startswith("model.encoder."):
                parameter_groups["encoder"]["params"].append(parameter)
            elif name.startswith("model.pointmap_head."):
                parameter_groups["pointmap"]["params"].append(parameter)
            elif name.startswith("model.decoder.final_refinement_head."):
                parameter_groups["refinement"]["params"].append(parameter)
            elif name.startswith("model.decoder."):
                parameter_groups["decoder"]["params"].append(parameter)
            else:
                parameter_groups["other"]["params"].append(parameter)

        optimizer_groups = [group for group in parameter_groups.values() if group["params"]]
        optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.args.max_epochs, 1),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @staticmethod
    def _namespace_to_dict(args) -> dict:
        if isinstance(args, Namespace):
            return vars(args)
        if isinstance(args, dict):
            return args
        return dict(vars(args))

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        args=None,
        config=None,
        experiment_logger=None,
        strict: bool = True,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hyper_parameters = checkpoint.get("hyper_parameters", {})

        if args is None:
            args = Namespace(**hyper_parameters.get("args", {}))
        if config is None:
            config = PCCRConfig(**hyper_parameters.get("config", {}))

        model = cls(args=args, config=config, experiment_logger=experiment_logger)
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        return model

    def on_save_checkpoint(self, checkpoint):
        checkpoint["hyper_parameters"] = {
            "args": self._namespace_to_dict(self.args),
            "config": self.config.to_dict(),
            "lr": self.lr,
        }
