import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import os
import json
from datetime import datetime
from pathlib import Path

class Callback:
    """Base callback class"""
    
    def __init__(self):
        pass
    
    def on_train_begin(self, **kwargs):
        pass
    
    def on_train_end(self, **kwargs):
        pass
    
    def on_epoch_begin(self, epoch: int, **kwargs):
        pass
    
    def on_epoch_end(self, epoch: int, **kwargs):
        pass
    
    def on_batch_begin(self, batch_idx: int, **kwargs):
        pass
    
    def on_batch_end(self, batch_idx: int, **kwargs):
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        super().__init__()
        
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_metric = float('inf') if mode == 'min' else -float('inf')
        self.best_epoch = 0
        
        if mode not in ['min', 'max']:
            raise ValueError(f"mode should be 'min' or 'max', got {mode}")
    
    def on_epoch_end(self, epoch: int, val_metrics: Dict, **kwargs) -> bool:
        """
        Check if training should stop
        
        Returns:
            bool: True if training should stop
        """
        if self.monitor not in val_metrics:
            print(f"Warning: Monitor {self.monitor} not in validation metrics")
            return False
        
        current = val_metrics[self.monitor]
        
        if self.mode == 'min':
            improvement = self.best_metric - current
        else:
            improvement = current - self.best_metric
        
        if improvement > self.min_delta:
            self.best_metric = current
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights and 'model' in kwargs:
                self.best_weights = kwargs['model'].state_dict().copy()
                print(f"New best {self.monitor}: {current:.4f} at epoch {epoch}")
        else:
            self.wait += 1
            print(f"Early stopping: {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                
                # Restore best weights
                if self.restore_best_weights and self.best_weights is not None:
                    kwargs['model'].load_state_dict(self.best_weights)
                    print(f"Restored best weights from epoch {self.best_epoch}")
                
                return True
        
        return False
    
    def on_train_end(self, **kwargs):
        if self.stopped_epoch > 0:
            print(f"Early stopping triggered at epoch {self.stopped_epoch}")


class ModelCheckpoint(Callback):
    """
    Model checkpoint callback
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        mode: str = 'min',
        save_freq: int = 1,
        max_save: int = 5
    ):
        super().__init__()
        
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.save_freq = save_freq
        self.max_save = max_save
        
        self.best_metric = float('inf') if mode == 'min' else -float('inf')
        self.saved_checkpoints = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        **kwargs
    ):
        # Check if should save
        if epoch % self.save_freq != 0 and self.save_best_only:
            return
        
        # Get current metric
        if self.monitor in val_metrics:
            current = val_metrics[self.monitor]
        elif self.monitor in train_metrics:
            current = train_metrics[self.monitor]
        else:
            print(f"Warning: Monitor {self.monitor} not found in metrics")
            return
        
        # Check if best
        if self.save_best_only:
            if self.mode == 'min':
                is_best = current < self.best_metric
            else:
                is_best = current > self.best_metric
            
            if not is_best:
                return
            
            self.best_metric = current
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_metric': self.best_metric
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add amp scaler if exists
        if 'scaler' in kwargs:
            checkpoint['scaler_state_dict'] = kwargs['scaler'].state_dict()
        
        # Save checkpoint
        if self.save_best_only:
            filename = f"best_{self.monitor}_{current:.4f}.pth"
        else:
            filename = f"checkpoint_epoch{epoch:03d}.pth"
        
        filepath = os.path.join(self.filepath, filename)
        torch.save(checkpoint, filepath)
        
        # Update saved checkpoints
        self.saved_checkpoints.append(filepath)
        
        # Remove old checkpoints if exceeds max_save
        if len(self.saved_checkpoints) > self.max_save:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
        
        print(f"Checkpoint saved: {filepath}")


class LearningRateScheduler(Callback):
    """
    Learning rate scheduler callback
    """
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        step_every: str = 'epoch'  # 'epoch' or 'batch'
    ):
        super().__init__()
        
        self.scheduler = scheduler
        self.step_every = step_every
        
        if step_every not in ['epoch', 'batch']:
            raise ValueError("step_every must be 'epoch' or 'batch'")
    
    def on_epoch_end(self, **kwargs):
        if self.step_every == 'epoch':
            self.scheduler.step()
    
    def on_batch_end(self, **kwargs):
        if self.step_every == 'batch':
            self.scheduler.step()


class TensorBoardLogger(Callback):
    """
    TensorBoard logging callback
    """
    
    def __init__(
        self,
        log_dir: str = './logs',
        update_freq: int = 10
    ):
        super().__init__()
        
        self.log_dir = log_dir
        self.update_freq = update_freq
        
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
        
        self.step = 0
    
    def on_batch_end(
        self,
        batch_idx: int,
        loss: float,
        outputs: Dict,
        targets: List,
        **kwargs
    ):
        if batch_idx % self.update_freq == 0:
            # Log training loss
            self.writer.add_scalar('train/loss', loss, self.step)
            
            # Log learning rate if available
            if 'optimizer' in kwargs:
                lr = kwargs['optimizer'].param_groups[0]['lr']
                self.writer.add_scalar('train/lr', lr, self.step)
            
            self.step += 1
    
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        **kwargs
    ):
        # Log training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        # Log validation metrics
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        # Log histograms of model parameters
        if 'model' in kwargs and epoch % 10 == 0:
            model = kwargs['model']
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(name, param, epoch)
    
    def on_train_end(self, **kwargs):
        self.writer.close()


class WandbLogger(Callback):
    """
    Weights & Biases logging callback
    """
    
    def __init__(
        self,
        project: str = 'riva-cell-detection',
        name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        super().__init__()
        
        try:
            import wandb
            self.wandb = wandb
            
            # Initialize wandb
            if self.wandb.run is None:
                self.wandb.init(project=project, name=name, config=config)
            
            self.step = 0
        
        except ImportError:
            print("Wandb not installed. Install with: pip install wandb")
            self.wandb = None
    
    def on_train_begin(self, **kwargs):
        if self.wandb:
            # Log model architecture
            if 'model' in kwargs:
                model = kwargs['model']
                self.wandb.watch(model, log='all', log_freq=100)
    
    def on_batch_end(
        self,
        batch_idx: int,
        loss: float,
        outputs: Dict,
        targets: List,
        **kwargs
    ):
        if self.wandb and batch_idx % 10 == 0:
            log_dict = {
                'train/batch_loss': loss,
                'train/step': self.step
            }
            
            # Log learning rate
            if 'optimizer' in kwargs:
                lr = kwargs['optimizer'].param_groups[0]['lr']
                log_dict['train/lr'] = lr
            
            self.wandb.log(log_dict, step=self.step)
            self.step += 1
    
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        **kwargs
    ):
        if self.wandb:
            # Prefix metrics
            train_log = {f'train/{k}': v for k, v in train_metrics.items()}
            val_log = {f'val/{k}': v for k, v in val_metrics.items()}
            
            # Combine
            log_dict = {**train_log, **val_log, 'epoch': epoch}
            
            self.wandb.log(log_dict, step=epoch)
    
    def on_train_end(self, **kwargs):
        if self.wandb:
            self.wandb.finish()


class EMA(Callback):
    """
    Exponential Moving Average callback
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.model = model
        self.decay = decay
        self.device = device
        
        # Create shadow model
        self.shadow = {}
        self.backup = {}
        
        # Register parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def on_batch_end(self, **kwargs):
        # Update shadow parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow parameters to model"""
        self.backup = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}


class GradientAccumulationCallback(Callback):
    """
    Gradient accumulation callback
    """
    
    def __init__(self, accumulation_steps: int = 1):
        super().__init__()
        
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def on_batch_begin(self, **kwargs):
        self.current_step += 1
    
    def on_batch_end(self, optimizer: torch.optim.Optimizer, **kwargs):
        if self.current_step % self.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


class SWA(Callback):
    """
    Stochastic Weight Averaging callback
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        swa_start: int = 10,
        swa_freq: int = 5,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.device = device
        
        self.swa_n = 0
        self.swa_model = None
    
    def on_epoch_end(self, epoch: int, **kwargs):
        if epoch >= self.swa_start and epoch % self.swa_freq == 0:
            self.update_swa()
    
    def update_swa(self):
        """Update SWA model"""
        if self.swa_model is None:
            # Initialize SWA model
            self.swa_model = {
                name: param.data.clone() for name, param in self.model.named_parameters()
            }
            self.swa_n = 1
        else:
            # Update SWA model
            for name, param in self.model.named_parameters():
                self.swa_model[name] = (
                    self.swa_model[name] * self.swa_n + param.data
                ) / (self.swa_n + 1)
            
            self.swa_n += 1
    
    def apply_swa(self):
        """Apply SWA parameters to model"""
        if self.swa_model is not None:
            for name, param in self.model.named_parameters():
                param.data = self.swa_model[name].clone()


class MixUpCutMix(Callback):
    """
    MixUp and CutMix augmentation callback
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        switch_prob: float = 0.5
    ):
        super().__init__()
        
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.switch_prob = switch_prob
        
        self.current_method = 'mixup' if np.random.random() < 0.5 else 'cutmix'
    
    def on_batch_begin(
        self,
        images: torch.Tensor,
        targets: List[Dict],
        **kwargs
    ):
        """Apply MixUp or CutMix"""
        batch_size = images.shape[0]
        
        if np.random.random() < self.switch_prob:
            self.current_method = 'mixup' if self.current_method == 'cutmix' else 'cutmix'
        
        if self.current_method == 'mixup':
            images, targets = self.mixup(images, targets, batch_size)
        else:
            images, targets = self.cutmix(images, targets, batch_size)
        
        return images, targets
    
    def mixup(
        self,
        images: torch.Tensor,
        targets: List[Dict],
        batch_size: int
    ):
        """Apply MixUp augmentation"""
        # Generate random permutation
        indices = torch.randperm(batch_size, device=images.device)
        
        # Mixup lambda
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Mix targets
        mixed_targets = []
        for i in range(batch_size):
            idx = indices[i]
            
            # Combine boxes from both samples
            boxes1 = targets[i]['boxes']
            boxes2 = targets[idx]['boxes']
            
            # Combine labels
            labels1 = targets[i]['labels']
            labels2 = targets[idx]['labels']
            
            # Mix with lambda
            if len(boxes1) > 0 and len(boxes2) > 0:
                # Randomly sample boxes from both
                n1 = int(len(boxes1) * lam)
                n2 = len(boxes2) - n1
                
                indices1 = torch.randperm(len(boxes1))[:n1]
                indices2 = torch.randperm(len(boxes2))[:n2]
                
                mixed_boxes = torch.cat([boxes1[indices1], boxes2[indices2]], dim=0)
                mixed_labels = torch.cat([labels1[indices1], labels2[indices2]], dim=0)
            elif len(boxes1) > 0:
                mixed_boxes = boxes1
                mixed_labels = labels1
            else:
                mixed_boxes = boxes2
                mixed_labels = labels2
            
            mixed_targets.append({
                'boxes': mixed_boxes,
                'labels': mixed_labels,
                'image_id': targets[i]['image_id'],
                'orig_size': targets[i]['orig_size'],
                'size': targets[i]['size']
            })
        
        return mixed_images, mixed_targets
    
    def cutmix(
        self,
        images: torch.Tensor,
        targets: List[Dict],
        batch_size: int
    ):
        """Apply CutMix augmentation"""
        # Generate random permutation
        indices = torch.randperm(batch_size, device=images.device)
        
        # CutMix parameters
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        cut_ratio = np.sqrt(1 - lam)
        
        # Image dimensions
        _, _, h, w = images.shape
        
        # Cut region
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Random position
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, x1 + cut_w)
        y2 = min(h, y1 + cut_h)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        
        # Mix targets (similar to mixup)
        mixed_targets = []
        for i in range(batch_size):
            idx = indices[i]
            
            boxes1 = targets[i]['boxes']
            boxes2 = targets[idx]['boxes']
            
            # Filter boxes that are in the cut region
            # This is simplified - in practice would need to check box intersection
            if len(boxes1) > 0 and len(boxes2) > 0:
                # Randomly sample based on lambda
                n1 = int(len(boxes1) * lam)
                n2 = len(boxes2) - n1
                
                indices1 = torch.randperm(len(boxes1))[:n1]
                indices2 = torch.randperm(len(boxes2))[:n2]
                
                mixed_boxes = torch.cat([boxes1[indices1], boxes2[indices2]], dim=0)
                mixed_labels = torch.cat([targets[i]['labels'][indices1], targets[idx]['labels'][indices2]], dim=0)
            elif len(boxes1) > 0:
                mixed_boxes = boxes1
                mixed_labels = targets[i]['labels']
            else:
                mixed_boxes = boxes2
                mixed_labels = targets[idx]['labels']
            
            mixed_targets.append({
                'boxes': mixed_boxes,
                'labels': mixed_labels,
                'image_id': targets[i]['image_id'],
                'orig_size': targets[i]['orig_size'],
                'size': targets[i]['size']
            })
        
        return mixed_images, mixed_targets