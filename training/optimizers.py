import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
    MultiStepLR,
    ExponentialLR
)
from typing import Dict, List, Optional, Tuple, Any
import math

def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'AdamW',
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    nesterov: bool = False,
    layerwise_lr: bool = False,
    backbone_lr_mult: float = 0.1,
    transformer_lr_mult: float = 1.0,
    head_lr_mult: float = 5.0
) -> torch.optim.Optimizer:
    """
    Create optimizer with optional layer-wise learning rates
    """
    
    if layerwise_lr:
        # Separate parameters by layer type
        backbone_params = []
        transformer_params = []
        head_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'backbone' in name.lower():
                backbone_params.append(param)
            elif 'transformer' in name.lower() or 'attention' in name.lower():
                transformer_params.append(param)
            elif 'head' in name.lower() or 'class' in name.lower() or 'box' in name.lower():
                head_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = []
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': lr * backbone_lr_mult,
                'weight_decay': weight_decay
            })
        
        if transformer_params:
            param_groups.append({
                'params': transformer_params,
                'lr': lr * transformer_lr_mult,
                'weight_decay': weight_decay
            })
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': lr * head_lr_mult,
                'weight_decay': weight_decay
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': lr,
                'weight_decay': weight_decay
            })
        
        print(f"Layer-wise learning rates:")
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            print(f"  LR={group['lr']:.2e}, Params={num_params:,}")
        
        params = param_groups
    else:
        # All parameters with same learning rate
        params = model.parameters()
    
    # Create optimizer
    if optimizer_name.lower() == 'adamw':
        optimizer = AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adam':
        optimizer = Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = SGD(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'CosineAnnealingWarmRestarts',
    epochs: int = 100,
    lr_min: float = 1e-6,
    warmup_epochs: int = 5,
    warmup_factor: float = 0.001,
    step_size: int = 30,
    gamma: float = 0.1,
    milestones: List[int] = None,
    T_0: int = 10,
    T_mult: int = 2,
    patience: int = 10,
    factor: float = 0.1,
    verbose: bool = True
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler
    """
    
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr_min
        )
    
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=lr_min
        )
    
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            verbose=verbose,
            min_lr=lr_min
        )
    
    elif scheduler_name == 'StepLR':
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    
    elif scheduler_name == 'MultiStepLR':
        if milestones is None:
            milestones = [epochs // 3, 2 * epochs // 3]
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
    
    elif scheduler_name == 'ExponentialLR':
        scheduler = ExponentialLR(
            optimizer,
            gamma=gamma
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    # Add warmup if specified
    if warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer,
            scheduler,
            warmup_epochs=warmup_epochs,
            warmup_factor=warmup_factor
        )
    
    return scheduler


class WarmupScheduler:
    """
    Warmup scheduler wrapper
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        warmup_epochs: int = 5,
        warmup_factor: float = 0.001
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.current_epoch = 0
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: int = None):
        if epoch is None:
            epoch = self.current_epoch + 1
        
        self.current_epoch = epoch
        
        if epoch <= self.warmup_epochs:
            # Warmup phase
            alpha = epoch / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        else:
            # Normal scheduler
            self.scheduler.step(epoch - self.warmup_epochs)
    
    def state_dict(self):
        return {
            'scheduler': self.scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'warmup_factor': self.warmup_factor,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_factor = state_dict['warmup_factor']
        self.base_lrs = state_dict['base_lrs']


class GradientAccumulator:
    """
    Gradient accumulation utility
    """
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_step(self) -> bool:
        """Check if optimizer should step"""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
    
    def reset(self):
        """Reset step counter"""
        self.current_step = 0


class Lookahead:
    """
    Lookahead optimizer wrapper
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        k: int = 5,
        alpha: float = 0.5
    ):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.counter = 0
        
        # Store slow weights
        self.slow_weights = []
        for group in optimizer.param_groups:
            for p in group['params']:
                self.slow_weights.append(p.data.clone())
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        
        self.counter += 1
        if self.counter >= self.k:
            # Update slow weights
            for i, (slow_p, fast_p) in enumerate(zip(self.slow_weights, self.optimizer.param_groups[0]['params'])):
                slow_p.add_(self.alpha * (fast_p.data - slow_p))
                fast_p.data.copy_(slow_p)
            
            self.counter = 0
        
        return loss
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class SAM:
    """
    Sharpness-Aware Minimization optimizer
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = False
    ):
        self.optimizer = optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.state = {}
    
    def step(self, closure):
        """Two-step SAM update"""
        # First step: compute gradient at current point
        loss = closure()
        loss.backward()
        
        # Store gradient and compute perturbation
        grads = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads.append(p.grad.data.clone())
        
        # Compute perturbation direction
        grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
        if self.adaptive:
            # Adaptive rho based on gradient norm
            scale = self.rho / (grad_norm + 1e-12)
        else:
            scale = self.rho
        
        # Apply perturbation
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(scale * p.grad.data)
        
        # Second step: compute gradient at perturbed point
        self.optimizer.zero_grad()
        loss = closure()
        loss.backward()
        
        # Restore original parameters and apply update
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.sub_(scale * p.grad.data)  # Restore
        
        # Apply update
        self.optimizer.step()
        
        return loss
    
    def zero_grad(self):
        self.optimizer.zero_grad()


class RAdam:
    """
    Rectified Adam optimizer
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=True
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.degenerated_to_sgd = degenerated_to_sgd
        
        self.state = {}
        self.t = 0
        
        # Initialize state
        for param in self.params:
            self.state[param] = {
                'step': 0,
                'exp_avg': torch.zeros_like(param.data),
                'exp_avg_sq': torch.zeros_like(param.data)
            }
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        self.t += 1
        
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            if self.weight_decay != 0:
                grad.add_(param.data, alpha=self.weight_decay)
            
            state = self.state[param]
            
            # Update biased first moment estimate
            state['exp_avg'].mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])
            
            # Update biased second raw moment estimate
            state['exp_avg_sq'].mul_(self.betas[1]).addcmul_(grad, grad, value=1 - self.betas[1])
            
            state['step'] += 1
            beta2_t = self.betas[1] ** state['step']
            N_sma_max = 2 / (1 - self.betas[1]) - 1
            N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
            
            # More conservative check
            if N_sma >= 5:
                # Compute bias-corrected second moment
                denom = state['exp_avg_sq'].sqrt().add_(self.eps)
                
                # Compute step size
                step_size = self.lr * math.sqrt(
                    (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) *
                    (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
                ) / (1 - self.betas[0] ** state['step'])
                
                # Update parameters
                param.data.addcdiv_(state['exp_avg'], denom, value=-step_size)
            else:
                # Degenerate to SGD
                step_size = self.lr / (1 - self.betas[0] ** state['step'])
                param.data.add_(state['exp_avg'], alpha=-step_size)
        
        return loss
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()