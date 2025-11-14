"""
Main Application Entry Point
AI Face Emotion & Persona Overlay - Real-time emotion detection with cyberpunk HUD.

Created by Ali Hamza & Zarmeena Jawad
"""
import cv2
import time
import sys
import traceback
import platform
from typing import Optional, Tuple
import numpy as np

from .camera import CameraManager, FaceDetector
from .models import EmotionModel
from .ui import (
    draw_rounded_rect,
    draw_scanline,
    draw_fps,
    draw_emotion_label,
    draw_glitch_text,
    draw_status_panel,
    save_screenshot,
    NEON_CYAN
)
from .utils.config import Config
from .utils.logger import get_logger

logger = get_logger(__name__)


def play_shutter_sound():
    """Play a short sound on screenshot (Windows only)."""
    try:
        if platform.system() == 'Windows':
            import winsound
            winsound.Beep(1000, 80)
            winsound.Beep(1400, 60)
    except Exception:
        pass


def main():
    """Main application loop."""
    config = Config()
    camera_manager = None
    detector = None
    emotion = None
    
    try:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  AI Face Emotion & Persona Overlay")
        logger.info("  Version 2.0.0")
        logger.info("=" * 60)
        logger.info("")
        
        # Initialize camera
        logger.info("Setting up camera...")
        camera_manager = CameraManager(config)
        if not camera_manager.open():
            logger.error("Failed to open camera. Please check:")
            logger.error("1. Camera permissions in System Settings")
            logger.error("2. No other apps are using the camera")
            logger.error("3. Camera is connected and working")
            return
        
        camera_info = camera_manager.get_info()
        logger.info(f"Camera opened: {camera_info.get('name', 'Unknown')} "
                   f"({camera_info.get('width')}x{camera_info.get('height')})")
        
        # Initialize face detector
        logger.info("Loading face detection model...")
        detector = FaceDetector(refine_landmarks=True, max_faces=1)
        
        # Initialize emotion model
        logger.info("Loading emotion detection model...")
        emotion = EmotionModel(config=config)
        emotion.dl_backend = config.get('model.dl_backend', 'onnx')
        try:
            emotion.load()
            logger.info("Emotion model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Hugging Face model: {e}")
            logger.info("Running in fallback mode (landmark-based detection)")
        
        logger.info("All systems ready!")
        logger.info("Starting webcam feed...")
        logger.info("")
        logger.info("Keyboard Controls:")
        logger.info("  ESC - Exit")
        logger.info("  S   - Save screenshot")
        logger.info("  +/- - Adjust smoothing")
        logger.info("  [/] - Adjust bbox speed")
        logger.info("  d   - Toggle deep learning")
        logger.info("  m   - Switch DL backend")
        logger.info("-" * 60)
        
        # Main loop variables
        fps_smooth = 30.0
        last_time = time.time()
        scan_y = 0
        display_bbox: Optional[Tuple[int, int, int, int]] = None
        
        # Smoothing controls
        label_decay = emotion.recent_decay
        bbox_lerp = emotion.bbox_lerp
        use_dl = False
        dl_backend = emotion.dl_backend
        
        frame_count = 0
        black_frame_warnings = 0
        
        consecutive_read_failures = 0
        max_consecutive_failures = 30  # Allow some failures during initialization
        
        while True:
            # Read frame
            ret, frame = camera_manager.read()
            if not ret:
                consecutive_read_failures += 1
                if consecutive_read_failures <= 5:
                    logger.debug(f"Frame read failed (attempt {consecutive_read_failures})")
                elif consecutive_read_failures == max_consecutive_failures:
                    logger.error("Too many consecutive frame read failures - camera may be disconnected")
                    break
                time.sleep(0.1)  # Brief delay before retry
                continue
            
            # Reset failure counter on successful read
            consecutive_read_failures = 0
            
            if frame is None or frame.size == 0:
                if frame_count < 10:
                    logger.warning("Received empty frame")
                continue
            
            # Check frame brightness (for Continuity Camera warm-up)
            frame_brightness = frame.mean()
            if frame_count < 30:
                if frame_brightness < 1.0:
                    black_frame_warnings += 1
                    if black_frame_warnings <= 3:
                        logger.info(f"Frame {frame_count}: Camera initializing...")
                elif black_frame_warnings > 0 and frame_brightness > 5.0:
                    logger.info(f"Frame {frame_count}: Camera feed active!")
                    black_frame_warnings = 0
                frame_count += 1
            
            # Skip black frames after warm-up
            if frame_count > 30 and frame_brightness < 1.0:
                continue
            
            h, w = frame.shape[:2]
            
            # Detect face (only process every N frames for performance if needed)
            # For now, process every frame for best responsiveness
            bbox, landmarks = detector.detect(frame)
            
            # Smooth bounding box movement
            if bbox:
                if display_bbox is None:
                    display_bbox = bbox
                else:
                    x0, y0, w0, h0 = display_bbox
                    x1, y1, w1, h1 = bbox
                    lerp = bbox_lerp
                    nx = int(x0 + (x1 - x0) * lerp)
                    ny = int(y0 + (y1 - y0) * lerp)
                    nw = int(w0 + (w1 - w0) * lerp)
                    nh = int(h0 + (h1 - h0) * lerp)
                    display_bbox = (nx, ny, nw, nh)
            else:
                display_bbox = None
            
            # Predict emotion (skip if no face detected for performance)
            label, conf, persona, alpha = ('neutral', 0.0, 'Calm Sentinel', 1.0)
            if bbox and emotion and landmarks:
                try:
                    if use_dl:
                        label, conf, persona, alpha = emotion.predict_dl(frame, bbox)
                    else:
                        label, conf, persona, alpha = emotion.predict(landmarks, (h, w))
                except Exception as e:
                    logger.debug(f"Emotion prediction failed: {e}, using neutral")
                    label, conf, persona, alpha = ('neutral', 0.5, 'Calm Sentinel', 1.0)
            
            # Draw overlays
            if display_bbox:
                frame = draw_rounded_rect(frame, display_bbox, NEON_CYAN, thickness=2, radius=22, glow=True)
                draw_emotion_label(frame, label, conf, persona, display_bbox, alpha)
            
            # Animated scanline
            scan_y = (scan_y + int((time.time() - last_time) * 180)) % h
            draw_scanline(frame, scan_y, color=NEON_CYAN, thickness=2)
            
            # Header text
            draw_glitch_text(frame, 'AI FACE EMOTION & PERSONA OVERLAY')
            
            # FPS counter
            now = time.time()
            fps = 1.0 / max(1e-6, (now - last_time))
            last_time = now
            fps_smooth = fps_smooth * 0.85 + fps * 0.15
            draw_fps(frame, fps_smooth)
            
            # Status panel
            status_lines = [
                f'Smoothing: {label_decay:.2f}  BBox lerp: {bbox_lerp:.2f}',
                f'DL: {"ON" if use_dl else "OFF"}  Backend: {dl_backend}',
                'Keys: +/- smoothing, [/] bbox, d toggle DL, m switch backend, s screenshot'
            ]
            draw_status_panel(frame, status_lines)
            
            # Display frame (only update if window exists for performance)
            cv2.imshow('AI Face Persona', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('=') or key == ord('+'):
                label_decay = min(0.98, label_decay + 0.03)
                emotion.recent_decay = label_decay
                logger.debug(f"Smoothing increased to {label_decay:.2f}")
            elif key == ord('-'):
                label_decay = max(0.5, label_decay - 0.03)
                emotion.recent_decay = label_decay
                logger.debug(f"Smoothing decreased to {label_decay:.2f}")
            elif key == ord(']'):
                bbox_lerp = min(0.9, bbox_lerp + 0.05)
                emotion.bbox_lerp = bbox_lerp
                logger.debug(f"BBox lerp increased to {bbox_lerp:.2f}")
            elif key == ord('['):
                bbox_lerp = max(0.02, bbox_lerp - 0.05)
                emotion.bbox_lerp = bbox_lerp
                logger.debug(f"BBox lerp decreased to {bbox_lerp:.2f}")
            elif key == ord('d'):
                use_dl = not use_dl
                logger.info(f"Deep learning mode: {'ON' if use_dl else 'OFF'}")
            elif key == ord('m'):
                dl_backend = 'onnx' if dl_backend == 'deepface' else 'deepface'
                emotion.dl_backend = dl_backend
                logger.info(f"DL backend switched to: {dl_backend}")
            elif key == ord('s') or key == ord('S'):
                fn = save_screenshot(frame)
                logger.info(f"Screenshot saved: {fn}")
                play_shutter_sound()
            elif key == 27:  # ESC
                logger.info("Exiting...")
                break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
    finally:
        if camera_manager:
            camera_manager.release()
        cv2.destroyAllWindows()
        logger.info("Application closed")


if __name__ == '__main__':
    main()

