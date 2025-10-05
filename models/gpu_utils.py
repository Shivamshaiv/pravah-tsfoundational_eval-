#!/usr/bin/env python3
"""
GPU Utilities for Model Training and Evaluation
Provides device detection, logging, and configuration
"""

import torch
import warnings
from typing import Dict, Optional


class GPUConfig:
    """GPU configuration and device management"""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.device = self._detect_device()
        self.device_name = self._get_device_name()
        
    def _detect_device(self) -> torch.device:
        """Detect best available device"""
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
    
    def _get_device_name(self) -> str:
        """Get device name for logging"""
        if self.cuda_available:
            return torch.cuda.get_device_name(0)
        else:
            return "CPU"
    
    def get_device_info(self) -> Dict[str, any]:
        """Get comprehensive device information"""
        info = {
            'device': str(self.device),
            'device_name': self.device_name,
            'cuda_available': self.cuda_available,
            'device_count': self.device_count,
            'pytorch_version': torch.__version__,
        }
        
        if self.cuda_available:
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = torch.backends.cudnn.version()
            info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0) / 1e9
        
        return info
    
    def print_device_info(self):
        """Print device information for logging"""
        print("=" * 60)
        print("GPU/DEVICE CONFIGURATION")
        print("=" * 60)
        info = self.get_device_info()
        print(f"PyTorch Version: {info['pytorch_version']}")
        print(f"Device: {info['device']}")
        print(f"Device Name: {info['device_name']}")
        print(f"CUDA Available: {info['cuda_available']}")
        
        if self.cuda_available:
            print(f"CUDA Version: {info['cuda_version']}")
            print(f"cuDNN Version: {info['cudnn_version']}")
            print(f"GPU Memory Total: {info['cuda_memory_total']:.2f} GB")
            print(f"GPU Memory Allocated: {info['cuda_memory_allocated']:.2f} GB")
            print(f"Number of GPUs: {info['device_count']}")
        else:
            print("\nWARNING: CUDA is not available. Training will use CPU.")
            print("To enable GPU acceleration, install NVIDIA drivers and CUDA toolkit.")
            print("See: https://pytorch.org/get-started/locally/")
        
        print("=" * 60)
        print()
    
    def get_trainer_kwargs(self, 
                          gradient_clip_val: float = 1.0,
                          precision: str = "32-true",
                          callbacks: Optional[list] = None) -> Dict:
        """
        Get PyTorch Lightning trainer kwargs optimized for device
        
        Args:
            gradient_clip_val: Gradient clipping value
            precision: Training precision ('32-true', '16-mixed', 'bf16-mixed')
            callbacks: List of Lightning callbacks
            
        Returns:
            Dictionary of trainer kwargs
        """
        trainer_kwargs = {
            "accelerator": "gpu" if self.cuda_available else "cpu",
            "devices": 1,
            "precision": precision,
            "gradient_clip_val": gradient_clip_val,
        }
        
        if callbacks:
            trainer_kwargs["callbacks"] = callbacks
        
        # Optimize for GPU if available
        if self.cuda_available:
            trainer_kwargs.update({
                "benchmark": True,  # cudnn benchmark for performance
            })
        
        return trainer_kwargs
    
    def optimize_for_device(self, batch_size: int = 32) -> int:
        """
        Suggest optimal batch size based on device
        
        Args:
            batch_size: Default batch size
            
        Returns:
            Optimized batch size
        """
        if not self.cuda_available:
            # Reduce batch size for CPU training
            return max(8, batch_size // 4)
        
        # For GPU, keep or increase batch size based on memory
        try:
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if mem_gb >= 40:  # A100 40GB
                return batch_size * 2
            elif mem_gb >= 24:  # RTX 3090/4090
                return int(batch_size * 1.5)
        except:
            pass
        
        return batch_size
    
    def clear_cache(self):
        """Clear GPU cache if CUDA is available"""
        if self.cuda_available:
            torch.cuda.empty_cache()
            print("GPU cache cleared")


# Global GPU configuration instance
_gpu_config = None


def get_gpu_config() -> GPUConfig:
    """Get or create global GPU configuration"""
    global _gpu_config
    if _gpu_config is None:
        _gpu_config = GPUConfig()
    return _gpu_config


def print_gpu_info():
    """Convenience function to print GPU information"""
    config = get_gpu_config()
    config.print_device_info()


def is_gpu_available() -> bool:
    """Check if GPU is available"""
    return torch.cuda.is_available()


def get_device() -> torch.device:
    """Get the default device"""
    config = get_gpu_config()
    return config.device


def get_trainer_kwargs(**kwargs) -> Dict:
    """Get optimized trainer kwargs"""
    config = get_gpu_config()
    return config.get_trainer_kwargs(**kwargs)


def optimize_batch_size(batch_size: int) -> int:
    """Optimize batch size for device"""
    config = get_gpu_config()
    return config.optimize_for_device(batch_size)


if __name__ == "__main__":
    # Test GPU configuration
    print_gpu_info()
    
    print(f"\nOptimized batch size (base=32): {optimize_batch_size(32)}")
    print(f"Optimized batch size (base=64): {optimize_batch_size(64)}")
    
    trainer_kwargs = get_trainer_kwargs()
    print(f"\nTrainer kwargs: {trainer_kwargs}")
