"""
Logging utilities for RIVA Cell Detection
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json


def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Get a configured logger
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        console: Whether to log to console
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class BaseLogger:
    """Base logger class for experiment tracking"""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        self.step = 0
        self.metrics_history = {}
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics"""
        if step is not None:
            self.step = step
        
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append({'step': self.step, 'value': value})
    
    def save(self):
        """Save metrics to file"""
        if self.log_dir:
            metrics_file = self.log_dir / 'metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)


class WandbLogger(BaseLogger):
    """
    Weights & Biases logger for experiment tracking
    """
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        log_dir: Optional[str] = None,
        enabled: bool = True
    ):
        super().__init__(log_dir)
        
        self.enabled = enabled
        self.run = None
        
        if enabled:
            try:
                import wandb
                self.wandb = wandb
                
                # Initialize wandb run
                self.run = wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    tags=tags,
                    dir=log_dir,
                    reinit=True
                )
                print(f"WandB initialized: {wandb.run.name}")
                
            except ImportError:
                print("Warning: wandb not installed. Logging disabled.")
                self.enabled = False
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb"""
        super().log(metrics, step)
        
        if self.enabled and self.wandb:
            self.wandb.log(metrics, step=step)
    
    def log_image(self, key: str, image, caption: Optional[str] = None):
        """Log image to wandb"""
        if self.enabled and self.wandb:
            self.wandb.log({key: self.wandb.Image(image, caption=caption)})
    
    def log_table(self, key: str, data: Dict[str, list]):
        """Log table to wandb"""
        if self.enabled and self.wandb:
            import pandas as pd
            df = pd.DataFrame(data)
            self.wandb.log({key: self.wandb.Table(dataframe=df)})
    
    def watch(self, model, log: str = 'gradients', log_freq: int = 100):
        """Watch model gradients"""
        if self.enabled and self.wandb:
            self.wandb.watch(model, log=log, log_freq=log_freq)
    
    def finish(self):
        """Finish wandb run"""
        super().save()
        
        if self.enabled and self.run:
            self.run.finish()
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.finish()
        except:
            pass


class TensorBoardLogger(BaseLogger):
    """
    TensorBoard logger for experiment tracking
    """
    
    def __init__(
        self,
        log_dir: str,
        enabled: bool = True
    ):
        super().__init__(log_dir)
        
        self.enabled = enabled
        self.writer = None
        
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"TensorBoard initialized: {log_dir}")
                
            except ImportError:
                print("Warning: tensorboard not installed. Logging disabled.")
                self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to tensorboard"""
        super().log(metrics, step)
        
        if self.enabled and self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
    
    def log_image(self, key: str, image, step: Optional[int] = None):
        """Log image to tensorboard"""
        if self.enabled and self.writer:
            self.writer.add_image(key, image, step, dataformats='HWC')
    
    def log_histogram(self, key: str, values, step: Optional[int] = None):
        """Log histogram to tensorboard"""
        if self.enabled and self.writer:
            self.writer.add_histogram(key, values, step)
    
    def close(self):
        """Close tensorboard writer"""
        super().save()
        
        if self.enabled and self.writer:
            self.writer.close()
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.close()
        except:
            pass


class MetricTracker:
    """
    Track and aggregate metrics during training
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self._metrics = {}
        self._counts = {}
    
    def update(self, metrics: Dict[str, float], n: int = 1):
        """
        Update tracked metrics
        
        Args:
            metrics: Dictionary of metric values
            n: Number of samples (for weighted averaging)
        """
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = 0.0
                self._counts[key] = 0
            
            self._metrics[key] += value * n
            self._counts[key] += n
    
    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics"""
        return {
            key: self._metrics[key] / self._counts[key]
            for key in self._metrics
            if self._counts[key] > 0
        }
    
    def __str__(self) -> str:
        """String representation of current metrics"""
        metrics = self.compute()
        return ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
