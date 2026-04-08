from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule

from src.loss import DiceScore
from src.pccr.config import PCCRConfig
from src.pccr.losses import RegistrationCriterion, SyntheticTargets
from src.pccr.model import PCCRModel
from src.pccr.utils import voxel_grid
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
            decoder_fitting_weight=config.decoder_fitting_weight,
            decoder_fitting_detach_target=config.decoder_fitting_detach_target,
            decoder_fitting_entropy_threshold=config.decoder_fitting_entropy_threshold,
            decoder_fitting_confidence_percentile=config.decoder_fitting_confidence_percentile,
            decoder_fitting_margin_power=config.decoder_fitting_margin_power,
            decoder_fitting_margin_min=config.decoder_fitting_margin_min,
            num_labels=config.num_labels,
        )
        self.experiment_logger = experiment_logger
        self.lr = args.lr
        self.test_outputs = []
        self._freeze_state = ""
        if getattr(self.args, "freeze_mode", "full") != "full":
            self._apply_explicit_freeze_mode()
        elif self.args.phase == "real" and self.config.refinement_warmup_epochs > 0:
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

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        oracle_canonical_source: torch.Tensor | None = None,
        oracle_canonical_target: torch.Tensor | None = None,
        oracle_dense_s2t: torch.Tensor | None = None,
        oracle_dense_t2s: torch.Tensor | None = None,
    ):
        return self.model(
            source,
            target,
            oracle_canonical_source=oracle_canonical_source,
            oracle_canonical_target=oracle_canonical_target,
            oracle_dense_s2t=oracle_dense_s2t,
            oracle_dense_t2s=oracle_dense_t2s,
        )

    @staticmethod
    def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if sigma <= 0:
            return torch.ones(1, device=device, dtype=dtype)
        radius = max(1, int(round(2.5 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.sum().clamp_min(1e-8)
        return kernel

    @classmethod
    def _gaussian_smooth_3d(cls, tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return tensor
        kernel_1d = cls._gaussian_kernel1d(sigma, tensor.device, tensor.dtype)
        radius = kernel_1d.numel() // 2
        smoothed = tensor
        for kernel_shape, padding in [
            ((kernel_1d.numel(), 1, 1), (0, 0, 0, 0, radius, radius)),
            ((1, kernel_1d.numel(), 1), (0, 0, radius, radius, 0, 0)),
            ((1, 1, kernel_1d.numel()), (radius, radius, 0, 0, 0, 0)),
        ]:
            weight = kernel_1d.view(1, 1, *kernel_shape).repeat(smoothed.shape[1], 1, 1, 1, 1)
            smoothed = F.pad(smoothed, padding, mode="replicate")
            smoothed = F.conv3d(smoothed, weight, groups=smoothed.shape[1])
        return smoothed

    @classmethod
    def _build_oracle_dense_displacement(
        cls,
        source_label: torch.Tensor,
        target_label: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        batch_size = source_label.shape[0]
        spatial_shape = tuple(source_label.shape[2:])
        grid = voxel_grid(spatial_shape, source_label.device).expand(batch_size, -1, -1, -1, -1)
        dense_fields = []
        for batch_index in range(batch_size):
            source_seg = source_label[batch_index, 0]
            target_seg = target_label[batch_index, 0]
            dense = source_label.new_zeros((3, *spatial_shape), dtype=torch.float32)
            support = source_label.new_zeros((1, *spatial_shape), dtype=torch.float32)
            labels = torch.unique(torch.cat([source_seg.reshape(-1), target_seg.reshape(-1)]))
            for label_value in labels.tolist():
                if int(label_value) == 0:
                    continue
                source_mask = source_seg == int(label_value)
                target_mask = target_seg == int(label_value)
                if not source_mask.any() or not target_mask.any():
                    continue
                source_center = grid[batch_index, :, source_mask].mean(dim=1)
                target_center = grid[batch_index, :, target_mask].mean(dim=1)
                delta = (target_center - source_center).to(dense.dtype)
                dense[:, source_mask] = delta.unsqueeze(-1)
                support[:, source_mask] = 1.0

            dense = dense.unsqueeze(0)
            support = support.unsqueeze(0)
            smoothed_dense = cls._gaussian_smooth_3d(dense * support, sigma=sigma)
            smoothed_support = cls._gaussian_smooth_3d(support, sigma=sigma)
            dense = smoothed_dense / smoothed_support.clamp_min(1e-4)
            dense = torch.where(smoothed_support > 1e-4, dense, torch.zeros_like(dense))
            dense_fields.append(dense.squeeze(0))
        return torch.stack(dense_fields, dim=0)

    def _compute_loss(self, batch):
        source, target, source_label, target_label = batch[:4]
        oracle_canonical_source = None
        oracle_canonical_target = None
        oracle_dense_s2t = None
        oracle_dense_t2s = None
        if getattr(self.args, "oracle_correspondence", "none") == "synthetic_gt" and len(batch) >= 6:
            oracle_canonical_source = batch[4]
            oracle_canonical_target = batch[5]
        if getattr(self.config, "diagnostic_oracle_handoff", False):
            oracle_dense_s2t = self._build_oracle_dense_displacement(
                source_label=source_label,
                target_label=target_label,
                sigma=self.config.oracle_handoff_gaussian_sigma,
            )
            oracle_dense_t2s = self._build_oracle_dense_displacement(
                source_label=target_label,
                target_label=source_label,
                sigma=self.config.oracle_handoff_gaussian_sigma,
            )
        outputs = self(
            source,
            target,
            oracle_canonical_source=oracle_canonical_source,
            oracle_canonical_target=oracle_canonical_target,
            oracle_dense_s2t=oracle_dense_s2t,
            oracle_dense_t2s=oracle_dense_t2s,
        )
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

    def _set_trainable_modules(self, *modules):
        self._set_requires_grad(self.model, False)
        for module in modules:
            if module is not None:
                self._set_requires_grad(module, True)

    def _apply_explicit_freeze_mode(self):
        freeze_mode = getattr(self.args, "freeze_mode", "full")
        if freeze_mode == "full":
            self._unfreeze_all_modules()
            return

        if freeze_mode == "final_refinement":
            self._set_trainable_modules(
                getattr(self.model.decoder, "final_refinement_head", None),
                getattr(self.model.decoder, "image_error_encoder", None),
                getattr(self.model.decoder, "local_cost_volume_encoder", None),
                getattr(self.model.decoder, "local_residual_matcher", None),
            )
        elif freeze_mode == "coarse_decoder":
            self._set_trainable_modules(self.model.decoder.stage_decoders)
        elif freeze_mode == "matcher":
            # Train the full correspondence branch, including any learned matcher refinement.
            self._set_trainable_modules(self.model.encoder, self.model.pointmap_head, self.model.matcher)
        elif freeze_mode == "decoder_and_refinement":
            self._set_trainable_modules(self.model.decoder)
        else:
            raise ValueError(f"Unsupported freeze mode: {freeze_mode}")

        self._freeze_state = freeze_mode

    def _apply_training_stage(self):
        if getattr(self.args, "freeze_mode", "full") != "full":
            if self._freeze_state != getattr(self.args, "freeze_mode", "full"):
                self._apply_explicit_freeze_mode()
            return
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
            "matcher": {"params": [], "lr": self.lr * self.config.canonical_head_lr_scale},
            "decoder": {"params": [], "lr": self.lr * self.config.coarse_decoder_lr_scale},
            "refinement": {"params": [], "lr": self.lr * self.config.residual_refinement_lr_scale},
            "other": {"params": [], "lr": self.lr},
        }

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("model.encoder."):
                parameter_groups["encoder"]["params"].append(parameter)
            elif name.startswith("model.pointmap_head."):
                parameter_groups["pointmap"]["params"].append(parameter)
            elif name.startswith("model.matcher."):
                parameter_groups["matcher"]["params"].append(parameter)
            elif (
                name.startswith("model.decoder.final_refinement_head.")
                or name.startswith("model.decoder.image_error_encoder.")
                or name.startswith("model.decoder.local_cost_volume_encoder.")
                or name.startswith("model.decoder.local_residual_matcher.")
            ):
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
        state_dict = checkpoint["state_dict"]
        if strict:
            model.load_state_dict(state_dict, strict=True)
            return model

        model_state = model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        for key, value in state_dict.items():
            if key not in model_state:
                skipped_keys.append(key)
                continue
            if model_state[key].shape != value.shape:
                skipped_keys.append(key)
                continue
            filtered_state_dict[key] = value

        incompatible = model.load_state_dict(filtered_state_dict, strict=False)
        if skipped_keys:
            print(
                f"[LiTPCCR] Skipped {len(skipped_keys)} incompatible checkpoint keys while loading "
                f"{checkpoint_path}."
            )
        if incompatible.missing_keys:
            print(f"[LiTPCCR] Missing keys after partial checkpoint load: {len(incompatible.missing_keys)}")
        return model

    def on_save_checkpoint(self, checkpoint):
        checkpoint["hyper_parameters"] = {
            "args": self._namespace_to_dict(self.args),
            "config": self.config.to_dict(),
            "lr": self.lr,
        }
