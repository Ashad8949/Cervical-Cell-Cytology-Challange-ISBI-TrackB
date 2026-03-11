"""
I/O utilities for RIVA Cell Detection
"""

import os
import yaml
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    best_metric: Optional[float] = None
):
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        save_path: Path to save checkpoint
        scheduler: Optional scheduler state
        scaler: Optional AMP scaler state
        metrics: Optional metrics dict
        config: Optional config dict
        best_metric: Optional best metric value
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if config is not None:
        checkpoint['config'] = config
    
    if best_metric is not None:
        checkpoint['best_metric'] = best_metric
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = 'cuda',
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load training checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        scaler: AMP scaler to load state into
        device: Device to load to
        strict: Whether to strictly enforce state dict keys
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"Model loaded from {checkpoint_path}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint


def generate_submission(
    predictions: List[Dict[str, np.ndarray]],
    image_filenames: List[str],
    output_path: str,
    image_sizes: Optional[List[tuple]] = None,
    confidence_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Generate submission CSV for RIVA competition
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        image_filenames: List of image filenames
        output_path: Path to save CSV
        image_sizes: Optional list of (H, W) tuples for denormalization
        confidence_threshold: Minimum confidence for predictions
    
    Returns:
        Submission DataFrame
    """
    rows = []
    
    for i, (pred, filename) in enumerate(zip(predictions, image_filenames)):
        boxes = pred.get('boxes', np.array([]))
        scores = pred.get('scores', np.array([]))
        labels = pred.get('labels', np.array([]))
        
        # Get image size for denormalization
        if image_sizes is not None:
            h, w = image_sizes[i]
        else:
            h, w = 1024, 1024  # Default assumption
        
        # Filter by confidence
        if len(scores) > 0:
            keep = scores >= confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        
        # Convert each box to submission format
        for j in range(len(boxes)):
            box = boxes[j]
            score = scores[j]
            label = labels[j] if len(labels) > j else 0
            
            # Convert from normalized cxcywh to absolute xywh
            if np.max(box) <= 1.0:
                cx, cy, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
            else:
                cx, cy, bw, bh = box[0], box[1], box[2], box[3]
            
            # Convert to x, y (top-left), width, height
            x = cx - bw / 2
            y = cy - bh / 2
            
            rows.append({
                'id': len(rows) + 1,
                'image_filename': filename,
                'class': int(label),
                'x': float(x),
                'y': float(y),
                'width': float(bw),
                'height': float(bh),
                'conf': float(score)
            })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure correct column order
    columns = ['id', 'image_filename', 'class', 'x', 'y', 'width', 'height', 'conf']
    df = df[columns]
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Submission saved: {output_path}")
    print(f"Total predictions: {len(df)}")
    print(f"Images with predictions: {df['image_filename'].nunique()}")
    
    return df


def load_annotations(csv_path: str) -> pd.DataFrame:
    """
    Load annotations from RIVA format CSV
    
    Args:
        csv_path: Path to annotations CSV
    
    Returns:
        DataFrame with annotations
    """
    df = pd.read_csv(csv_path)
    
    # Ensure expected columns exist
    expected_columns = ['image_filename', 'x', 'y', 'width', 'height']
    for col in expected_columns:
        if col not in df.columns:
            # Try alternatives
            alt_cols = {
                'image_filename': ['filename', 'image', 'file'],
                'x': ['x_center', 'cx', 'center_x'],
                'y': ['y_center', 'cy', 'center_y'],
                'width': ['w', 'box_width'],
                'height': ['h', 'box_height']
            }
            
            for alt in alt_cols.get(col, []):
                if alt in df.columns:
                    df[col] = df[alt]
                    break
    
    return df


def get_image_paths(
    image_dir: str,
    extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
) -> List[Path]:
    """
    Get all image paths from directory
    
    Args:
        image_dir: Directory containing images
        extensions: List of valid image extensions
    
    Returns:
        List of image paths
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.glob(f'*{ext}'))
        image_paths.extend(image_dir.glob(f'*{ext.upper()}'))
    
    return sorted(image_paths)


def setup_experiment_dir(
    base_dir: str,
    experiment_name: str
) -> Dict[str, Path]:
    """
    Setup experiment directory structure
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
    
    Returns:
        Dictionary with paths to subdirectories
    """
    base_dir = Path(base_dir)
    exp_dir = base_dir / experiment_name
    
    dirs = {
        'root': exp_dir,
        'checkpoints': exp_dir / 'checkpoints',
        'logs': exp_dir / 'logs',
        'visualizations': exp_dir / 'visualizations',
        'predictions': exp_dir / 'predictions'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs
