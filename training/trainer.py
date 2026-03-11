import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional, Callable, Any
import time
import os
import warnings
warnings.filterwarnings('ignore')

class Trainer:
    """
    Main trainer class for cell detection models
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        criterion: nn.Module,
        device: str = 'cuda',
        config: Dict[str, Any] = None,
        callbacks: List[Callable] = None,
        logger: Optional[Any] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config or {}
        self.callbacks = callbacks or []
        self.logger = logger
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf')
        self.train_history = []
        self.val_history = []
        
        # Mixed precision
        self.use_amp = self.config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.accumulation_steps = self.config.get('accumulation_steps', 1)
        
        # Logging
        self.log_interval = self.config.get('log_interval', 10)
        self.save_dir = self.config.get('save_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        print(f"Trainer initialized on {self.device}")
        print(f"Using mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.accumulation_steps}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device, non_blocking=True)
            targets = self._prepare_targets(targets)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                # Only sum actual losses (exclude diagnostic metrics)
                loss = sum(v for k, v in loss_dict.items() if k.startswith('loss_'))
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('grad_clip_norm', 0.1)
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update scheduler
                if self.scheduler is not None and self.config.get('scheduler_type') == 'step':
                    self.scheduler.step()
                
                # Update global step
                self.global_step += 1
            
            # Update loss and metrics
            epoch_loss += loss.item() * self.accumulation_steps
            
            # Update metrics
            for k, v in loss_dict.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v.item()
            
            # Logging
            if batch_idx % self.log_interval == 0:
                current_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
                # Log to wandb/tensorboard
                if self.logger is not None:
                    log_dict = {
                        'train/loss': current_loss,
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/grad_norm': grad_norm if 'grad_norm' in locals() else 0
                    }
                    self.logger.log(log_dict, step=self.global_step)
            
            # Callbacks
            for callback in self.callbacks:
                callback.on_batch_end(
                    batch_idx=batch_idx,
                    loss=loss.item(),
                    outputs=outputs,
                    targets=targets
                )
        
        # Average metrics
        num_batches = len(self.train_loader)
        epoch_loss /= num_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        return {'loss': epoch_loss, **epoch_metrics}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        val_metrics = {}
        
        # For detection metrics
        from utils.metrics import DetectionMetrics
        evaluator = DetectionMetrics(iou_threshold=0.5)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for images, targets in pbar:
                images = images.to(self.device, non_blocking=True)
                prepared_targets = self._prepare_targets(targets)
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, prepared_targets)
                    loss = sum(v for k, v in loss_dict.items() if k.startswith('loss_'))
                
                val_loss += loss.item()
                
                # Update metrics
                for k, v in loss_dict.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v.item()
                
                # Collect predictions for mAP calculation
                predictions = self._post_process(outputs)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                
                pbar.set_postfix({'val_loss': f'{val_loss/(pbar.n+1):.4f}'})
        
        # Average validation metrics
        num_batches = len(self.val_loader)
        val_loss /= num_batches
        for k in val_metrics:
            val_metrics[k] /= num_batches
        
        # Compute detection metrics
        det_metrics = evaluator.compute(all_predictions, all_targets)
        
        return {
            'val_loss': val_loss,
            **val_metrics,
            **det_metrics
        }
    
    def fit(self, epochs: int) -> Dict[str, List[float]]:
        """Main training loop"""
        print(f"Starting training for {epochs} epochs")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mAP': [],
            'val_precision': [],
            'val_recall': []
        }
        
        # Callbacks on training start
        for callback in self.callbacks:
            callback.on_train_begin()
        
        try:
            for epoch in range(epochs):
                self.epoch = epoch + 1
                
                # Callbacks on epoch begin
                for callback in self.callbacks:
                    callback.on_epoch_begin(self.epoch)
                
                # Train
                train_metrics = self.train_epoch()
                history['train_loss'].append(train_metrics['loss'])
                
                # Validate
                if self.epoch % self.config.get('eval_every', 1) == 0:
                    val_metrics = self.validate()
                    history['val_loss'].append(val_metrics['val_loss'])
                    history['val_mAP'].append(val_metrics.get('mAP', 0))
                    history['val_precision'].append(val_metrics.get('precision', 0))
                    history['val_recall'].append(val_metrics.get('recall', 0))
                    
                    # Update best metric
                    current_metric = val_metrics.get('mAP', -val_metrics['val_loss'])
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self._save_checkpoint('best.pth', val_metrics)
                    
                    # Log validation results
                    print(f"\nEpoch {self.epoch} Validation:")
                    for k, v in val_metrics.items():
                        if 'loss' in k or 'mAP' in k or 'precision' in k or 'recall' in k:
                            print(f"  {k}: {v:.4f}")
                    
                    if self.logger is not None:
                        log_dict = {f'val/{k}': v for k, v in val_metrics.items()}
                        self.logger.log(log_dict, step=self.global_step)
                
                # Update scheduler (epoch-based)
                if self.scheduler is not None and self.config.get('scheduler_type') == 'epoch':
                    self.scheduler.step()
                
                # Save checkpoint
                if self.epoch % self.config.get('save_every', 10) == 0:
                    self._save_checkpoint(f'checkpoint_epoch{self.epoch}.pth', val_metrics)
                
                # Callbacks on epoch end
                for callback in self.callbacks:
                    should_stop = callback.on_epoch_end(
                        epoch=self.epoch,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics
                    )
                    if should_stop:
                        print(f"Early stopping at epoch {self.epoch}")
                        break
                
                # Early stopping check
                if hasattr(self, 'should_stop') and self.should_stop:
                    break
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Callbacks on training end
            for callback in self.callbacks:
                callback.on_train_end()
            
            # Save final model
            self._save_checkpoint('final.pth', {})
        
        return history
    
    def _prepare_targets(self, targets: List[Dict]) -> List[Dict]:
        """Prepare targets for loss computation"""
        prepared = []
        for target in targets:
            prepared_target = {
                'labels': target['labels'].to(self.device),
                'boxes': target['boxes'].to(self.device),
                'image_id': target.get('image_id', torch.tensor([0]).to(self.device)),
                'orig_size': target.get('orig_size', torch.tensor([1024, 1024]).to(self.device)),
                'size': target.get('size', torch.tensor([1024, 1024]).to(self.device))
            }
            prepared.append(prepared_target)
        return prepared
    
    def _post_process(self, outputs: Dict[str, torch.Tensor]) -> List[Dict]:
        """Post-process model outputs"""
        predictions = []
        
        # Apply softmax to logits
        logits = outputs['pred_logits'].softmax(-1)
        boxes = outputs['pred_boxes']
        
        batch_size = logits.shape[0]
        
        for i in range(batch_size):
            # Filter by confidence
            scores, labels = logits[i][:, :-1].max(dim=-1)
            keep = scores > self.config.get('confidence_threshold', 0.1)
            
            prediction = {
                'boxes': boxes[i][keep].cpu().numpy(),
                'scores': scores[keep].cpu().numpy(),
                'labels': labels[keep].cpu().numpy()
            }
            predictions.append(prediction)
        
        return predictions
    
    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config,
            'metrics': metrics
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', -float('inf'))
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, global step {self.global_step}")
        print(f"Best metric: {self.best_metric:.4f}")