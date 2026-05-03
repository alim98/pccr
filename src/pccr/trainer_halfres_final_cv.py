from __future__ import annotations

from src.pccr.model_halfres_final_cv import PCCRModelHalfResFinalCV
from src.pccr.trainer import LiTPCCR


class LiTPCCRHalfResFinalCV(LiTPCCR):
    """Isolated Lightning module that swaps in the half-resolution final-CV model."""

    def __init__(self, args, config, experiment_logger=None):
        super().__init__(args=args, config=config, experiment_logger=experiment_logger)
        self.model = PCCRModelHalfResFinalCV(config)
        self._freeze_state = ""
        if getattr(self.args, "freeze_mode", "full") != "full":
            self._apply_explicit_freeze_mode()
        elif self.args.phase == "real" and self.config.refinement_warmup_epochs > 0:
            self._freeze_encoder_and_canonical_heads()
