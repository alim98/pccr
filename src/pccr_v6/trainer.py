from __future__ import annotations

from argparse import Namespace

import torch

from src.pccr.trainer import LiTPCCR
from src.pccr_v6.config import PCCRV6Config
from src.pccr_v6.model import PCCRV6Model


class LiTPCCRV6(LiTPCCR):
    """v6 trainer wrapper that reuses the stable training/eval pipeline."""

    def __init__(self, args, config: PCCRV6Config, experiment_logger=None):
        super().__init__(args=args, config=config, experiment_logger=experiment_logger)
        self.config = config
        self.model = PCCRV6Model(config)
        self._freeze_state = ""
        if getattr(self.args, "freeze_mode", "full") != "full":
            self._apply_explicit_freeze_mode()
        elif self.args.phase == "real" and self.config.refinement_warmup_epochs > 0:
            self._freeze_encoder_and_canonical_heads()

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
                f"[LiTPCCRV6] Skipped {len(skipped_keys)} incompatible checkpoint keys while loading "
                f"{checkpoint_path}."
            )
        if incompatible.missing_keys:
            print(f"[LiTPCCRV6] Missing keys after partial checkpoint load: {len(incompatible.missing_keys)}")
        return model
