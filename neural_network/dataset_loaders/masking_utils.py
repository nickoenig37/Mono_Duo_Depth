"""
Advanced masking utilities for depth estimation training.
Handles RealSense-specific issues and stereo artifacts.
"""

import torch
import numpy as np
from typing import Tuple, Optional

class DepthMaskGenerator:
    """
    Generates validity masks for depth/disparity data.
    
    Handles common issues in RealSense data:
    - Distance limitations (>10m unreliable)
    - Reflective surfaces (windows, mirrors)
    - Edge artifacts
    - Stereo occlusions
    """
    
    def __init__(
        self,
        max_distance: float = 10.0,  # meters
        min_distance: float = 0.3,   # meters (RealSense minimum)
        edge_margin: int = 20,        # pixels to exclude from edges
        occlusion_threshold: float = 1.0,  # disparity jump threshold
        outlier_percentile: float = 99.5,  # for detecting extreme outliers
        baseline: float = 0.1371,    # stereo baseline in meters (from your calib)
        focal_length: float = 719.53  # focal length in pixels (from your calib)
    ):
        """
        Args:
            max_distance: Maximum reliable depth (meters). RealSense D435 ~10m
            min_distance: Minimum reliable depth (meters). RealSense D435 ~0.3m
            edge_margin: Pixels to exclude from image borders
            occlusion_threshold: Disparity gradient threshold for occlusion detection
            outlier_percentile: Percentile threshold for outlier detection
            baseline: Stereo camera baseline (meters)
            focal_length: Camera focal length (pixels)
        """
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.edge_margin = edge_margin
        self.occlusion_threshold = occlusion_threshold
        self.outlier_percentile = outlier_percentile
        self.baseline = baseline
        self.focal_length = focal_length
        
        # Convert distance limits to disparity limits
        # disparity = baseline * focal_length / depth
        self.min_disparity = (baseline * focal_length) / max_distance
        self.max_disparity = (baseline * focal_length) / min_distance
        
        print(f"  Min disparity (at {max_distance}m): {self.min_disparity:.2f} pixels")
        print(f"  Max disparity (at {min_distance}m): {self.max_disparity:.2f} pixels")
    
    def create_basic_mask(self, disparity: torch.Tensor) -> torch.Tensor:
        """
        Basic validity mask: finite values only.
        
        Args:
            disparity: (B, H, W) or (B, 1, H, W) disparity tensor
            
        Returns:
            Boolean mask of same shape
        """
        return torch.isfinite(disparity)
    
    def create_range_mask(self, disparity: torch.Tensor) -> torch.Tensor:
        """
        Mask based on valid disparity/distance range.
        
        Filters out:
        - Distances > 10m (unreliable RealSense measurements)
        - Distances < 0.3m (too close for RealSense)
        - Zero/negative disparities (invalid)
        
        Args:
            disparity: (B, H, W) or (B, 1, H, W) disparity tensor
            
        Returns:
            Boolean mask of same shape
        """
        valid_range = (disparity > self.min_disparity) & (disparity < self.max_disparity)
        valid_positive = disparity > 0  # Disparity must be positive
        
        return valid_range & valid_positive
    
    def create_edge_mask(
        self, 
        disparity: torch.Tensor,
        margin: Optional[int] = None
    ) -> torch.Tensor:
        """
        Mask excluding image edges where stereo matching is unreliable.
        
        Why edge masking?
        - Stereo matching needs surrounding context
        - Edge pixels have incomplete matching windows
        - Rectification distortion highest at edges
        - Lens distortion artifacts at borders
        
        Args:
            disparity: (B, H, W) or (B, 1, H, W) disparity tensor
            margin: Override default edge margin
            
        Returns:
            Boolean mask of same shape
        """
        if margin is None:
            margin = self.edge_margin
        
        mask = torch.ones_like(disparity, dtype=torch.bool)
        
        # Handle both (B, H, W) and (B, 1, H, W) inputs
        if disparity.dim() == 4:
            _, _, H, W = disparity.shape
            mask[:, :, :margin, :] = False      # Top edge
            mask[:, :, -margin:, :] = False     # Bottom edge
            mask[:, :, :, :margin] = False      # Left edge
            mask[:, :, :, -margin:] = False     # Right edge
        elif disparity.dim() == 3:
            _, H, W = disparity.shape
            mask[:, :margin, :] = False      # Top edge
            mask[:, -margin:, :] = False     # Bottom edge
            mask[:, :, :margin] = False      # Left edge
            mask[:, :, -margin:] = False     # Right edge
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {disparity.dim()}D")
        
        return mask
    
    def create_occlusion_mask(
        self, 
        disparity: torch.Tensor,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Detect occluded regions using disparity discontinuities.
        
        How occlusion masking works:
        - At object boundaries, disparity changes sharply
        - Large gradient magnitude = likely occlusion
        - Mask these unreliable regions
        
        Args:
            disparity: (B, H, W) or (B, 1, H, W) disparity tensor
            threshold: Gradient magnitude threshold (disparity pixels)
            
        Returns:
            Boolean mask: False where occlusion detected
        """
        if threshold is None:
            threshold = self.occlusion_threshold
        
        # Ensure 4D: (B, 1, H, W)
        if disparity.dim() == 3:
            disparity = disparity.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Compute gradients using Sobel-like filters
        # Gradient in x direction (horizontal disparity changes)
        grad_x = torch.nn.functional.conv2d(
            disparity,
            weight=torch.tensor([[[[-1., 0., 1.]]]], device=disparity.device, dtype=disparity.dtype),
            padding=(0, 1)
        )
        
        # Gradient in y direction (vertical disparity changes)
        grad_y = torch.nn.functional.conv2d(
            disparity,
            weight=torch.tensor([[[[-1.], [0.], [1.]]]], device=disparity.device, dtype=disparity.dtype),
            padding=(1, 0)
        )
        
        # Gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Mask where gradient is below threshold (smooth regions = reliable)
        occlusion_mask = grad_magnitude < threshold
        
        # Return to original shape if needed
        if squeeze_output:
            occlusion_mask = occlusion_mask.squeeze(1)
        
        return occlusion_mask
    
    def create_outlier_mask(
        self, 
        disparity: torch.Tensor,
        percentile: Optional[float] = None
    ) -> torch.Tensor:
        """
        Detect outliers (e.g., window reflections, IR interference).
        
        Why outlier masking?
        - Windows reflect IR patterns -> incorrect depth
        - Shiny surfaces cause IR scattering -> spikes
        - RealSense occasionally produces random bad pixels
        
        Detection method:
        - Compute local statistics in each image
        - Mark values beyond Nth percentile as outliers
        - More robust than simple thresholding
        
        Args:
            disparity: (B, H, W) or (B, 1, H, W) disparity tensor
            percentile: Percentile threshold (default 99.5 = top 0.5% are outliers)
            
        Returns:
            Boolean mask: False where outliers detected
        """
        if percentile is None:
            percentile = self.outlier_percentile
        
        # Store original shape
        original_shape = disparity.shape
        
        # Handle both (B, H, W) and (B, 1, H, W)
        if disparity.dim() == 4:
            B, C, H, W = disparity.shape
            squeeze_output = False
        elif disparity.dim() == 3:
            disparity = disparity.unsqueeze(1)
            B, C, H, W = disparity.shape
            squeeze_output = True
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {disparity.dim()}D")
        
        outlier_mask = torch.ones_like(disparity, dtype=torch.bool)
        
        for b in range(B):
            for c in range(C):
                disp_slice = disparity[b, c]
                
                # Only consider finite values for statistics
                finite_mask = torch.isfinite(disp_slice)
                if finite_mask.sum() == 0:
                    continue  # All values invalid
                
                valid_disparities = disp_slice[finite_mask]
                
                # Compute upper threshold (e.g., 99.5th percentile)
                upper_threshold = torch.quantile(valid_disparities, percentile / 100.0)
                
                # Also check for extreme low values (could be reflections showing "infinite" depth)
                lower_threshold = torch.quantile(valid_disparities, (100 - percentile) / 100.0)
                
                # Mark outliers
                is_outlier = (disp_slice > upper_threshold) | (disp_slice < lower_threshold)
                outlier_mask[b, c][is_outlier] = False
        
        # Return to original shape
        if squeeze_output:
            outlier_mask = outlier_mask.squeeze(1)
        
        return outlier_mask
    
    def create_combined_mask(
        self,
        disparity: torch.Tensor,
        use_basic: bool = True,
        use_range: bool = True,
        use_edge: bool = True,
        use_occlusion: bool = True,
        use_outlier: bool = True,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Create comprehensive validity mask combining all techniques.
        
        Args:
            disparity: (B, H, W) or (B, 1, H, W) disparity tensor
            use_basic: Apply finite value check
            use_range: Apply distance range filtering
            use_edge: Apply edge masking
            use_occlusion: Apply occlusion detection
            use_outlier: Apply outlier detection
            verbose: Print statistics about masked pixels
            
        Returns:
            combined_mask: Boolean tensor, True = valid pixel
            stats: Dictionary with masking statistics
        """
        stats = {}
        total_pixels = disparity.numel()
        
        # Start with all pixels valid
        combined_mask = torch.ones_like(disparity, dtype=torch.bool)
        
        # Apply each mask type
        if use_basic:
            basic_mask = self.create_basic_mask(disparity)
            combined_mask &= basic_mask
            stats['invalid_values'] = total_pixels - basic_mask.sum().item()
        
        if use_range:
            range_mask = self.create_range_mask(disparity)
            combined_mask &= range_mask
            stats['out_of_range'] = total_pixels - range_mask.sum().item()
        
        if use_edge:
            edge_mask = self.create_edge_mask(disparity)
            combined_mask &= edge_mask
            stats['edge_pixels'] = total_pixels - edge_mask.sum().item()
        
        if use_occlusion:
            occlusion_mask = self.create_occlusion_mask(disparity)
            combined_mask &= occlusion_mask
            stats['occluded_pixels'] = total_pixels - occlusion_mask.sum().item()
        
        if use_outlier:
            outlier_mask = self.create_outlier_mask(disparity)
            combined_mask &= outlier_mask
            stats['outlier_pixels'] = total_pixels - outlier_mask.sum().item()
        
        # Final statistics
        valid_pixels = combined_mask.sum().item()
        stats['valid_pixels'] = valid_pixels
        stats['total_pixels'] = total_pixels
        stats['valid_percentage'] = 100.0 * valid_pixels / total_pixels
        
        if verbose:
            print(f"\n{'='*60}")
            print("Masking Statistics:")
            print(f"{'='*60}")
            print(f"Total pixels: {total_pixels:,}")
            print(f"Valid pixels: {valid_pixels:,} ({stats['valid_percentage']:.1f}%)")
            print(f"\nMasked pixels breakdown:")
            if 'invalid_values' in stats:
                print(f"  Invalid values (NaN/inf): {stats['invalid_values']:,}")
            if 'out_of_range' in stats:
                print(f"  Out of range (>10m or <0.3m): {stats['out_of_range']:,}")
            if 'edge_pixels' in stats:
                print(f"  Edge regions: {stats['edge_pixels']:,}")
            if 'occluded_pixels' in stats:
                print(f"  Occlusions: {stats['occluded_pixels']:,}")
            if 'outlier_pixels' in stats:
                print(f"  Outliers (reflections/noise): {stats['outlier_pixels']:,}")
            print(f"{'='*60}\n")
        
        return combined_mask, stats
