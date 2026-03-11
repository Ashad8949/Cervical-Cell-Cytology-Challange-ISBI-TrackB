import cv2
import numpy as np
from typing import Tuple, Optional, List
from skimage import exposure, filters

def stain_normalization(
    image: np.ndarray,
    target_mean: Optional[np.ndarray] = None,
    target_std: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Normalize stain appearance in histopathology images
    
    Args:
        image: Input RGB image
        target_mean: Target mean values for each channel
        target_std: Target std values for each channel
    
    Returns:
        Normalized image
    """
    # Convert to float
    image = image.astype(np.float32) / 255.0
    
    # Default targets (ImageNet statistics)
    if target_mean is None:
        target_mean = np.array([0.485, 0.456, 0.406])
    if target_std is None:
        target_std = np.array([0.229, 0.224, 0.225])
    
    # Normalize each channel
    for i in range(3):
        image[..., i] = (image[..., i] - target_mean[i]) / target_std[i]
    
    # Clip and rescale to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def remove_timestamps(image: np.ndarray) -> np.ndarray:
    """
    Remove timestamp artifacts from images
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define orange color range for timestamps
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Dilate mask to cover text edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Inpaint if timestamp detected
    if np.sum(mask) > 0:
        image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    return image


def clean_artifacts(
    image: np.ndarray,
    min_size: int = 100
) -> np.ndarray:
    """
    Remove small artifacts from images
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Remove small contours
    mask = np.ones_like(gray) * 255
    for contour in contours:
        if cv2.contourArea(contour) < min_size:
            cv2.drawContours(mask, [contour], 0, 0, -1)
    
    # Apply mask
    result = image.copy()
    result[mask == 0] = 0
    
    return result


def normalize_intensity(
    image: np.ndarray,
    method: str = 'histogram'
) -> np.ndarray:
    """
    Normalize image intensity
    """
    if method == 'histogram':
        # Histogram equalization
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    
    elif method == 'clahe':
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])
        image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    
    elif method == 'adaptive':
        # Adaptive histogram equalization
        image = exposure.equalize_adapthist(
            image, clip_limit=0.03, kernel_size=None
        )
        image = (image * 255).astype(np.uint8)
    
    return image


def extract_patch(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 10
) -> np.ndarray:
    """
    Extract patch around bounding box with padding
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]


def compute_cell_density(
    image: np.ndarray,
    bboxes: List[Tuple[float, float, float, float]]
) -> float:
    """
    Compute cell density in image
    """
    if len(bboxes) == 0:
        return 0.0
    
    h, w = image.shape[:2]
    image_area = h * w
    
    # Compute total area covered by cells
    cell_area = 0
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cell_area += (x2 - x1) * (y2 - y1)
    
    return cell_area / image_area


def create_cell_mask(
    image_shape: Tuple[int, int],
    bboxes: List[Tuple[float, float, float, float]]
) -> np.ndarray:
    """
    Create binary mask for cells
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    return mask