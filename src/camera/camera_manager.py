"""
Camera Manager
Handles camera initialization, device enumeration, and frame capture.
Prioritizes Mac built-in camera over Continuity Camera.

Created by Ali Hamza & Zarmeena Jawad
"""
import cv2
import platform
import time
from typing import Optional, Tuple, List, Dict
import numpy as np
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CameraManager:
    """Manages camera device selection and initialization."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize camera manager.
        
        Args:
            config: Configuration instance (uses singleton if None)
        """
        self.config = config or Config()
        self.cap: Optional[cv2.VideoCapture] = None
        self.device_index: Optional[int] = None
        self.device_name: Optional[str] = None
        self.is_macos = platform.system() == 'Darwin'
        self.backend = cv2.CAP_AVFOUNDATION if self.is_macos else cv2.CAP_ANY
        
    def _get_available_cameras(self) -> List[Dict[str, any]]:
        """Enumerate available camera devices.
        
        Returns:
            List of camera info dicts with 'index', 'name', 'is_builtin', 'is_continuity'
        """
        cameras = []
        
        # Suppress OpenCV warnings during enumeration
        import logging
        import os
        import sys
        opencv_logger = logging.getLogger('cv2')
        original_level = opencv_logger.level
        opencv_logger.setLevel(logging.ERROR)
        
        devnull = None
        try:
            # Try to enumerate cameras - limit to reasonable range
            # Most systems have 0-2 cameras, check up to 5 to be safe
            max_cameras = 5
            devnull = open(os.devnull, 'w')
            
            for idx in range(max_cameras):
                try:
                    # Suppress stderr for this operation
                    old_stderr = sys.stderr
                    sys.stderr = devnull
                    
                    try:
                        test_cap = cv2.VideoCapture(idx, self.backend)
                    finally:
                        sys.stderr = old_stderr
                    
                    if test_cap.isOpened():
                        # Try to read a frame to verify it's actually working
                        ret, test_frame = test_cap.read()
                        if ret and test_frame is not None:
                            # Try to get device name (may not work on all systems)
                            device_name = "Unknown"
                            try:
                                # On macOS, try to get device name
                                if self.is_macos:
                                    # AVFoundation may provide device info
                                    device_name = f"Camera {idx}"
                            except Exception:
                                pass
                            
                            # Check if it's a built-in camera (heuristic: usually index 0 on Mac)
                            is_builtin = idx == 0 and self.is_macos
                            # Check if it's Continuity Camera (heuristic: usually has "iPhone" or "Continuity" in name)
                            is_continuity = "iPhone" in device_name or "Continuity" in device_name
                            
                            cameras.append({
                                'index': idx,
                                'name': device_name,
                                'is_builtin': is_builtin,
                                'is_continuity': is_continuity
                            })
                        test_cap.release()
                except Exception:
                    continue
        finally:
            if devnull is not None:
                devnull.close()
            # Restore original logging level
            opencv_logger.setLevel(original_level)
        
        return cameras
    
    def _select_best_camera(self, cameras: List[Dict[str, any]]) -> Optional[int]:
        """Select the best camera based on preferences.
        
        Args:
            cameras: List of available camera info dicts
            
        Returns:
            Selected camera index or None
        """
        if not cameras:
            return None
        
        prefer_builtin = self.config.get('camera.prefer_builtin', True)
        skip_continuity = self.config.get('camera.skip_continuity', True)
        
        # Priority order:
        # 1. Built-in Mac camera (if prefer_builtin)
        # 2. Any non-Continuity camera
        # 3. Continuity camera (if not skipping)
        # 4. Any camera
        
        if prefer_builtin:
            for cam in cameras:
                if cam['is_builtin']:
                    logger.info(f"Selected built-in camera: index {cam['index']}")
                    return cam['index']
        
        # Skip Continuity Camera if configured
        if skip_continuity:
            for cam in cameras:
                if not cam['is_continuity']:
                    logger.info(f"Selected camera: index {cam['index']} ({cam['name']})")
                    return cam['index']
        
        # Fallback to first available
        if cameras:
            logger.info(f"Selected camera: index {cameras[0]['index']} ({cameras[0]['name']})")
            return cameras[0]['index']
        
        return None
    
    def open(self) -> bool:
        """Open the best available camera.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        logger.info("Enumerating available cameras...")
        cameras = self._get_available_cameras()
        
        if not cameras:
            logger.error("No cameras found")
            return False
        
        logger.info(f"Found {len(cameras)} camera(s)")
        for cam in cameras:
            logger.info(f"  - Index {cam['index']}: {cam['name']} (builtin={cam['is_builtin']}, continuity={cam['is_continuity']})")
        
        selected_idx = self._select_best_camera(cameras)
        if selected_idx is None:
            logger.error("Could not select a camera")
            return False
        
        self.device_index = selected_idx
        self.device_name = next((c['name'] for c in cameras if c['index'] == selected_idx), "Unknown")
        
        logger.info(f"Opening camera {selected_idx}...")
        self.cap = cv2.VideoCapture(selected_idx, self.backend)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {selected_idx}")
            return False
        
        # Set camera properties
        width = self.config.get('camera.width', 1280)
        height = self.config.get('camera.height', 720)
        fps = self.config.get('camera.fps', 30)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Warm up camera with better retry logic
        warmup_frames = self.config.get('camera.warmup_frames', 15)
        logger.info(f"Warming up camera ({warmup_frames} frames)...")
        
        valid_frames = 0
        consecutive_failures = 0
        max_failures = 5
        
        for i in range(warmup_frames * 3):  # Try up to 3x warmup frames
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                consecutive_failures = 0
                # Check if frame has content (not all black)
                frame_brightness = frame.mean()
                if frame_brightness > 1.0:
                    valid_frames += 1
                    if valid_frames >= 3:  # Need at least 3 valid frames
                        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        logger.info(f"Camera ready: {actual_width}x{actual_height}")
                        return True
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.warning(f"Camera read failed {consecutive_failures} times - retrying...")
                    # Try to reinitialize
                    self.cap.release()
                    time.sleep(0.5)
                    self.cap = cv2.VideoCapture(self.device_index, self.backend)
                    if not self.cap.isOpened():
                        logger.error("Failed to reopen camera after read failures")
                        return False
                    consecutive_failures = 0
            
            time.sleep(0.05)  # Shorter delay for faster warm-up
        
        if valid_frames < 3:
            logger.warning("Camera opened but frames appear black - will continue and retry during operation")
            # Still return True, frames may start coming through during operation
        
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera with retry logic.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        
        # Check if camera is still opened
        if not self.cap.isOpened():
            # Try to reopen
            logger.warning("Camera connection lost, attempting to reopen...")
            self.cap.release()
            time.sleep(0.2)
            self.cap = cv2.VideoCapture(self.device_index, self.backend)
            if not self.cap.isOpened():
                logger.error("Failed to reopen camera")
                return False, None
        
        # Try to read frame with retry
        max_retries = 3
        for attempt in range(max_retries):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                return True, frame
            
            if attempt < max_retries - 1:
                time.sleep(0.05)  # Brief delay before retry
        
        # If all retries failed, try to check camera status
        if not self.cap.isOpened():
            logger.warning("Camera closed during read")
            return False, None
        
        return False, None
    
    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")
    
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()
    
    def get_info(self) -> Dict[str, any]:
        """Get camera information.
        
        Returns:
            Dict with camera info
        """
        if not self.is_opened():
            return {}
        
        return {
            'index': self.device_index,
            'name': self.device_name,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
        }

