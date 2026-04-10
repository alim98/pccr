import math
import warnings
from argparse import Namespace
from contextlib import nullcontext

import torch
from lightning import LightningModule

from src import logger, checkpoints_dir
from src.model.hvit import HViT
from src.model.hvit_light import HViT_Light
from src.loss import loss_functions, DiceScore, DiceLoss
from src.utils import get_one_hot

dtype_map = {
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
    'fp16': torch.float16
}

class LiTHViT(LightningModule):
    def __init__(self, args, config, experiment_logger=None, save_model_every_n_epochs=10):
        super().__init__()
        self.automatic_optimization = False
        self.args = args
        self.config = config
        self.best_val_loss = 1e8
        self.save_model_every_n_epochs = save_model_every_n_epochs
        self.lr = args.lr
        self.last_epoch = 0
        self.tgt2src_reg = args.tgt2src_reg
        self.hvit_light = args.hvit_light
        self.precision = args.precision

        self.hvit = HViT_Light(config) if self.hvit_light else HViT(config)

        self.loss_weights = {
            "mse": self.args.mse_weights,
            "dice": self.args.dice_weights,
            "grad": self.args.grad_weights
        }
        self.dice_loss = DiceLoss(num_class=self.args.num_labels)
        self.experiment_logger = experiment_logger
        self.test_step_outputs = []

    def _resolve_eval_label_ids(self):
        label_ids = getattr(self.args, "eval_label_ids", None)
        if not label_ids:
            label_ids = self.config.get("eval_label_ids", [])
        return [
            int(label_id)
            for label_id in label_ids
            if 0 <= int(label_id) < self.args.num_labels
        ]

    def _mean_eval_score(self, score):
        if not isinstance(score, torch.Tensor):
            return torch.tensor(float(score), device=self.device)

        label_ids = self._resolve_eval_label_ids()
        if label_ids:
            return score[:, label_ids].mean()
        if score.shape[1] > 1:
            return score[:, 1:].mean()
        return score.mean()

    def _autocast_context(self):
        device_type = self.device.type if self.device is not None else "cpu"
        dtype_ = dtype_map.get(self.precision, torch.float32)

        if dtype_ == torch.float32:
            return nullcontext()
        if device_type == "cpu" and dtype_ != torch.bfloat16:
            return nullcontext()
        return torch.amp.autocast(device_type=device_type, dtype=dtype_)

    def _log_metrics(self, metrics, step=None):
        if self.experiment_logger is None:
            return
        self.experiment_logger.log_metrics(metrics, step=step)

    def _forward(self, batch, calc_score: bool = False, tgt2src_reg: bool = False):
        _loss = {}
        _score = 0.


        dtype_ = dtype_map.get(self.precision, torch.float32)

        with self._autocast_context():
            if tgt2src_reg:
                target, source = batch[0].to(dtype=dtype_), batch[1].to(dtype=dtype_)
                tgt_seg, src_seg = batch[2], batch[3]
            else:
                source, target = batch[0].to(dtype=dtype_), batch[1].to(dtype=dtype_)
                src_seg, tgt_seg = batch[2], batch[3]
                
            moved, flow = self.hvit(source, target)

            if calc_score:
                moved_seg = self._get_one_hot_from_src(src_seg, flow, self.args.num_labels)
                _score = DiceScore(moved_seg, tgt_seg.long(), self.args.num_labels)

            _loss = {}
            for key, weight in self.loss_weights.items():
                if key == "mse":
                    _loss[key] = weight * loss_functions[key](moved, target)
                elif key == "dice":
                    moved_seg = self._get_one_hot_from_src(src_seg, flow, self.args.num_labels)
                    _loss[key] = weight * self.dice_loss(moved_seg, tgt_seg.long())
                elif key == "grad":
                    _loss[key] = weight * loss_functions[key](flow)
            
            _loss["avg_loss"] = sum(_loss.values()) / len(_loss)
        return _loss, _score

    def training_step(self, batch, batch_idx):
        self.hvit.train()
        opt = self.optimizers()
        
        loss1, _ = self._forward(batch, calc_score=False)
        self.manual_backward(loss1["avg_loss"])
        opt.step()
        opt.zero_grad()
            
        if self.tgt2src_reg:
            loss2, _ = self._forward(batch, tgt2src_reg=True, calc_score=False)
            self.manual_backward(loss2["avg_loss"])
            opt.step()
            opt.zero_grad()
        
        total_loss = {
            key: (loss1[key].item() + loss2[key].item()) / 2 if self.tgt2src_reg and key in loss2 else loss1[key].item()
            for key in loss1.keys()
        }

        self._log_metrics(total_loss, step=self.global_step)
        return total_loss

    def on_train_epoch_end(self):
        if self.current_epoch % self.save_model_every_n_epochs == 0:
            checkpoint_path = f"{checkpoints_dir}/model_epoch_{self.current_epoch}.ckpt"
            self.trainer.save_checkpoint(checkpoint_path)
            logger.info(f"Saved model at epoch {self.current_epoch}")
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self._log_metrics({"learning_rate": current_lr}, step=self.global_step)


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.hvit.eval()
            _loss, _score = self._forward(batch, calc_score=True)
        score_mean = self._mean_eval_score(_score)
    
        # Log each component of the validation loss
        for loss_name, loss_value in _loss.items():
            self.log(f"val_{loss_name}", loss_value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    
        # Log the mean validation score if available
        if _score is not None:
            self.log("val_score", score_mean, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    
        # Log to the configured experiment tracker
        log_dict = {f"val_{k}": v.item() for k, v in _loss.items()}
        log_dict.update({
            "val_score_mean": score_mean.item() if _score is not None else None,
        })
        self._log_metrics({k: v for k, v in log_dict.items() if v is not None}, step=self.global_step)
    
        return {"val_loss": _loss["avg_loss"], "val_score": score_mean.item()}

    def on_validation_epoch_end(self):
        """
        Callback method called at the end of the validation epoch.
        Saves the best model based on validation loss and logs metrics.
        """
        val_loss = self.trainer.callback_metrics.get("val_loss")
        
        if val_loss is not None and self.current_epoch > 0:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = f"{checkpoints_dir}/best_model.ckpt"
                self.trainer.save_checkpoint(best_model_path)
                self._log_metrics({
                    "best_model_saved": 1.0,
                    "best_val_loss": self.best_val_loss.item()
                }, step=self.global_step)
                logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step on a batch of data.
        
        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.
        
        Returns:
            A dictionary containing the test Dice score.
        """
        with torch.no_grad():
            self.hvit.eval()
            _, _score = self._forward(batch, calc_score=True)
    
        # Ensure _score is a tensor and take the mean
        _score = self._mean_eval_score(_score)
    
        self.test_step_outputs.append(_score)   

        # Log to the configured experiment tracker only if available
        self._log_metrics({"test_dice": _score.item()}, step=self.global_step)

        # Return as a dict with tensor values
        return {"test_dice": _score}
    
    def on_test_epoch_end(self):
        """
        Callback method called at the end of the test epoch.
        Computes and logs the average test Dice score.
        """
        # Calculate the average Dice score across all test steps
        avg_test_dice = torch.stack(self.test_step_outputs).mean()

        # Log the average test Dice score
        self.log("avg_test_dice", avg_test_dice, prog_bar=True)

        # Log to the configured experiment tracker if available
        self._log_metrics({"total_test_dice_avg": avg_test_dice.item()})

        # Clear the test step outputs list for the next test epoch
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for the model.
        
        Returns:
            A dictionary containing the optimizer and learning rate scheduler configuration.
        """
        optimizer = torch.optim.Adam(self.hvit.parameters(), lr=self.lr, weight_decay=0, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def lr_lambda(self, epoch):
        """
        Defines the learning rate schedule.
        
        Args:
            epoch: The current epoch number.
        
        Returns:
            The learning rate multiplier for the given epoch.
        """
        max_epochs = self.trainer.max_epochs
        return math.pow(1 - epoch / max_epochs, 0.9)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, args=None, experiment_logger=None):
        """
        Loads a model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
            args: Optional arguments to override saved ones.
            experiment_logger: Optional experiment logger instance.
        
        Returns:
            An instance of the model loaded from the checkpoint.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        args = args or checkpoint.get('hyper_parameters', {}).get('args')
        if isinstance(args, dict):
            args = Namespace(**args)
        config = checkpoint.get('hyper_parameters', {}).get('config')
        
        save_every = getattr(args, "save_model_every_n_epochs", 10) if args is not None else 10
        model = cls(args, config, experiment_logger, save_model_every_n_epochs=save_every)
        model.load_state_dict(checkpoint['state_dict'])

        if 'hyper_parameters' in checkpoint:
            hyper_params = checkpoint['hyper_parameters']
            for attr in ['lr', 'best_val_loss', 'last_epoch']:
                setattr(model, attr, hyper_params.get(attr, getattr(model, attr)))

        return model

    def on_save_checkpoint(self, checkpoint):
        """
        Callback to save additional information in the checkpoint.
        
        Args:
            checkpoint: The checkpoint dictionary to be saved.
        """
        checkpoint['hyper_parameters'] = {
            'args': vars(self.args) if hasattr(self.args, "__dict__") else self.args,
            'config': self.config,
            'lr': self.lr,
            'best_val_loss': self.best_val_loss,
            'last_epoch': self.current_epoch
        }

    def _get_one_hot_from_src(self, src_seg, flow, num_labels):
        """
        Converts source segmentation to one-hot encoding and applies deformation.
        
        Args:
            src_seg: Source segmentation.
            flow: Deformation flow.
            num_labels: Number of segmentation labels.
        
        Returns:
            Deformed one-hot encoded segmentation.
        """
        src_seg_onehot = get_one_hot(src_seg, self.args.num_labels)
        deformed_segs = [
            self.hvit.spatial_trans(src_seg_onehot[:, i:i+1, ...].float(), flow.float())
            for i in range(num_labels)
        ]
        return torch.cat(deformed_segs, dim=1)
