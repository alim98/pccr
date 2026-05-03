from __future__ import annotations

from argparse import Namespace

import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from src.pccr.trainer import LiTPCCR
from src.pccr_v6.config import PCCRV6Config
from src.pccr_v6.losses import RegistrationCriterionV6
from src.pccr_v6.model import PCCRV6Model


class LiTPCCRV6(LiTPCCR):
    """v6 trainer: v6 model + full-res multiscale GDL criterion."""

    def __init__(self, args, config: PCCRV6Config, experiment_logger=None):
        super().__init__(args=args, config=config, experiment_logger=experiment_logger)
        self.config = config
        self.model = PCCRV6Model(config)
        self.criterion = RegistrationCriterionV6(
            image_size=config.data_size,
            phase=args.phase,
            image_loss_weight=config.image_loss_weight,
            multiscale_similarity_factors=config.multiscale_similarity_factors,
            multiscale_similarity_weights=config.multiscale_similarity_weights,
            lncc_window_size=config.lncc_window_size,
            segmentation_supervision_weight=config.segmentation_supervision_weight,
            smoothness_weight=config.smoothness_weight,
            jacobian_weight=config.jacobian_weight,
            inverse_consistency_weight=config.inverse_consistency_weight,
            correspondence_weight=config.correspondence_weight,
            synthetic_matchability_weight=config.synthetic_matchability_weight,
            residual_velocity_weight=config.residual_velocity_weight,
            decoder_fitting_weight=config.decoder_fitting_weight,
            decoder_fitting_detach_target=config.decoder_fitting_detach_target,
            decoder_fitting_entropy_threshold=config.decoder_fitting_entropy_threshold,
            decoder_fitting_confidence_percentile=config.decoder_fitting_confidence_percentile,
            decoder_fitting_margin_power=config.decoder_fitting_margin_power,
            decoder_fitting_margin_min=config.decoder_fitting_margin_min,
            num_labels=config.num_labels,
        )
        self._freeze_state = ""
        if getattr(self.args, "freeze_mode", "full") != "full":
            self._apply_explicit_freeze_mode()
        elif self.args.phase == "real" and self.config.refinement_warmup_epochs > 0:
            self._freeze_encoder_and_canonical_heads()
        self._update_hot_stages()

    def _update_hot_stages(self) -> None:
        new_stage_ids = frozenset(getattr(self.config, "new_stage_ids", None) or [])
        warmup = getattr(self.config, "new_stage_warmup_epochs", 0)
        all_stages = frozenset(self.config.pointmap_stage_ids)
        if warmup > 0 and self.current_epoch < warmup:
            hot = all_stages - new_stage_ids
        else:
            hot = all_stages
        # fall back to all stages if hot is empty (e.g. no warm stages exist)
        if not hot:
            hot = all_stages
        if hasattr(self.criterion, "set_hot_stages"):
            self.criterion.set_hot_stages(hot)
            cold = all_stages - hot
            if hasattr(self.criterion, "set_cold_stages"):
                self.criterion.set_cold_stages(cold)
            if new_stage_ids and self.args.phase == "real":
                if cold:
                    rank_zero_info(
                        f"[LiTPCCRV6] Stage warmup: cold={sorted(cold)}, "
                        f"hot={sorted(hot)}, epoch={self.current_epoch}/{warmup}"
                    )

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._update_hot_stages()

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
            config = PCCRV6Config(**hyper_parameters.get("config", {}))

        model = cls(args=args, config=config, experiment_logger=experiment_logger)
        state_dict = cls._remap_legacy_checkpoint_keys(checkpoint["state_dict"])
        if strict:
            model.load_state_dict(state_dict, strict=True)
            return model

        model_state = model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        partial_keys = []
        for key, value in state_dict.items():
            if key not in model_state:
                skipped_keys.append(key)
                continue
            if model_state[key].shape != value.shape:
                if value.ndim == model_state[key].ndim and value.is_floating_point() and model_state[key].is_floating_point():
                    merged_value = model_state[key].clone()
                    slices = tuple(slice(0, min(src_dim, dst_dim)) for src_dim, dst_dim in zip(value.shape, merged_value.shape))
                    merged_value[slices] = value[slices].to(dtype=merged_value.dtype)
                    filtered_state_dict[key] = merged_value
                    partial_keys.append(key)
                else:
                    skipped_keys.append(key)
                continue
            filtered_state_dict[key] = value

        incompatible = model.load_state_dict(filtered_state_dict, strict=False)
        if skipped_keys:
            rank_zero_info(
                f"[LiTPCCRV6] Skipped {len(skipped_keys)} incompatible checkpoint keys while loading "
                f"{checkpoint_path}."
            )
        if partial_keys:
            rank_zero_info(
                f"[LiTPCCRV6] Partially loaded {len(partial_keys)} resized checkpoint tensors "
                f"from {checkpoint_path}."
            )
        if incompatible.missing_keys:
            rank_zero_info(f"[LiTPCCRV6] Missing keys after partial checkpoint load: {len(incompatible.missing_keys)}")
        return model
