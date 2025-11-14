"""
Overlay Utilities Module
Modern cyberpunk HUD overlay rendering with improved visual effects.

Created by Ali Hamza & Zarmeena Jawad
"""
import cv2
import numpy as np
import time
import os
from typing import Tuple, List, Optional
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Load colors from config (with defaults)
_config = Config()
_color_config = _config.get('ui.colors', {})
NEON_CYAN = tuple(_color_config.get('neon_cyan', [200, 220, 255]))  # BGR
NEON_BLUE = tuple(_color_config.get('neon_blue', [160, 90, 255]))
NEON_ACCENT = tuple(_color_config.get('neon_accent', [180, 140, 255]))


def draw_rounded_rect(img: np.ndarray, rect: Tuple[int, int, int, int], color: Tuple[int, int, int], 
                      thickness: int = 2, radius: int = 22, glow: bool = True) -> np.ndarray:
    """Draw a rounded rectangle with enhanced glow effect.

    Args:
        img: Image to draw on
        rect: (x, y, w, h) bounding box
        color: BGR color tuple
        thickness: Line thickness
        radius: Corner radius
        glow: Enable enhanced glow effect

    Returns:
        Modified image
    """
    x, y, w, h = rect
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    if glow:
        # Enhanced multi-layer glow effect - only in the bbox area
        mask = np.zeros_like(img)
        cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
        # Multiple blur layers for smoother glow - blend only the glow area
        for k, alpha in ((35, 0.15), (25, 0.12), (15, 0.08), (7, 0.04)):
            blurred = cv2.GaussianBlur(mask, (k*2+1, k*2+1), 0)
            # Only blend where there's glow, preserve original frame elsewhere
            mask_area = (blurred > 0).any(axis=2, keepdims=True)
            img = np.where(mask_area, cv2.addWeighted(img, 1.0, blurred, alpha, 0), img)

    # Draw border directly on the frame without blending
    cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)
    
    return img


def draw_scanline(img: np.ndarray, y_pos: int, color: Optional[Tuple[int, int, int]] = None, 
                  thickness: int = 2) -> np.ndarray:
    """Draw animated scanline with enhanced glow effect.
    
    Args:
        img: Image to draw on
        y_pos: Y position of scanline
        color: BGR color (uses NEON_CYAN if None)
        thickness: Line thickness
        
    Returns:
        Modified image
    """
    h, w = img.shape[:2]
    if color is None:
        color = NEON_CYAN
    
    # Draw glow layers first (faint, behind main line)
    for offset, alpha in [(-3, 0.3), (-2, 0.2), (-1, 0.15), (1, 0.15), (2, 0.1)]:
        glow_color = tuple(max(0, min(255, int(c * alpha))) for c in color)
        if 0 <= y_pos + offset < h:
            cv2.line(img, (0, y_pos + offset), (w, y_pos + offset), glow_color, thickness)
    
    # Main scanline drawn directly
    cv2.line(img, (0, y_pos), (w, y_pos), color, thickness)
    return img


def draw_fps(img: np.ndarray, fps: float, pos: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Draw FPS counter with enhanced styling.
    
    Args:
        img: Image to draw on
        fps: FPS value
        pos: Position tuple (uses config if None)
        
    Returns:
        Modified image
    """
    if pos is None:
        pos_config = _config.get('ui.fps_position', [10, 28])
        pos = tuple(pos_config)
    
    text = f"FPS: {int(fps)}"
    # Shadow for better visibility
    cv2.putText(img, text, (pos[0]+2, pos[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
    # Main text
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.85, NEON_BLUE, 2, cv2.LINE_AA)
    return img


def draw_emotion_label(img: np.ndarray, label: str, conf: float, persona: str, 
                       bbox: Tuple[int, int, int, int], alpha: float = 1.0) -> np.ndarray:
    """Draw emotion label and persona with enhanced styling and confidence bar.
    
    Args:
        img: Image to draw on
        label: Emotion label
        conf: Confidence (0-1)
        persona: Persona name
        bbox: Face bounding box (x, y, w, h)
        alpha: Fade alpha (0-1)
        
    Returns:
        Modified image
    """
    x, y, w, h = bbox
    # Position above the bbox with better spacing
    px, py = x, max(15, y - 50)
    
    # Emotion label text
    label_text = f"{label.upper()}"
    conf_text = f"{int(conf*100)}%"
    
    # Glitch effect during fade-in
    jitter_x = int(3 * (1 - alpha))
    offset = (jitter_x, 0)
    
    # Draw label with shadow
    cv2.putText(img, label_text, (px+3+offset[0], py+3), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
    color = NEON_CYAN if alpha > 0.5 else NEON_BLUE
    cv2.putText(img, label_text, (px+offset[0], py), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    
    # Confidence percentage
    conf_y = py + 28
    cv2.putText(img, conf_text, (px+3+offset[0], conf_y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, conf_text, (px+offset[0], conf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, NEON_ACCENT, 2, cv2.LINE_AA)
    
    # Confidence bar
    bar_width = 120
    bar_height = 4
    bar_x = px
    bar_y = conf_y + 20
    bar_fill = int(bar_width * conf)
    
    # Bar background
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (30, 30, 30), -1)
    # Bar fill with gradient effect
    if bar_fill > 0:
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_fill, bar_y + bar_height), color, -1)
        # Subtle glow on bar - only in the bar area
        glow_mask = np.zeros_like(img)
        cv2.rectangle(glow_mask, (bar_x, bar_y-1), (bar_x + bar_fill, bar_y + bar_height+1), color, -1)
        blurred = cv2.GaussianBlur(glow_mask, (5, 5), 0)
        # Only blend where there's glow
        glow_area = (blurred > 0).any(axis=2, keepdims=True)
        img = np.where(glow_area, cv2.addWeighted(img, 1.0, blurred, 0.3, 0), img)
    
    # Persona below
    persona_y = bar_y + 25
    cv2.putText(img, persona, (px+3+offset[0], persona_y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, persona, (px+offset[0], persona_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, NEON_CYAN, 2, cv2.LINE_AA)
    
    return img


def draw_glitch_text(img: np.ndarray, text: str, pos: Optional[Tuple[int, int]] = None, 
                     base_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """Draw glitch-style text with enhanced multi-layer effect.
    
    Args:
        img: Image to draw on
        text: Text to display
        pos: Position tuple (uses config if None)
        base_color: BGR color (uses NEON_ACCENT if None)
        
    Returns:
        Modified image
    """
    if pos is None:
        pos_config = _config.get('ui.header_position', [20, 40])
        pos = tuple(pos_config)
    if base_color is None:
        base_color = NEON_ACCENT
    
    x, y = pos
    # Enhanced multi-layer glitch effect
    # Shadow
    cv2.putText(img, text, (x-2, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 8, cv2.LINE_AA)
    # Glitch layers with different colors
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 150), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x-2, y-2), cv2.FONT_HERSHEY_DUPLEX, 1.0, (100, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x+1, y-1), cv2.FONT_HERSHEY_DUPLEX, 1.0, (150, 255, 100), 2, cv2.LINE_AA)
    # Main text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, base_color, 3, cv2.LINE_AA)
    return img


def draw_status_panel(img: np.ndarray, lines: List[str], pos: Optional[Tuple[int, int]] = None, 
                      bg_color: Tuple[int, int, int] = (10, 10, 20), alpha: float = 0.7) -> np.ndarray:
    """Draw an enhanced translucent status panel with better styling.
    
    Args:
        img: Image to draw on
        lines: List of text lines to display
        pos: Position tuple (uses config if None)
        bg_color: Background color
        alpha: Transparency (0-1)
        
    Returns:
        Modified image
    """
    if pos is None:
        pos_config = _config.get('ui.status_panel_position', [10, 60])
        pos = tuple(pos_config)
    
    x, y = pos
    h, w = img.shape[:2]
    panel_w = 380
    panel_h = 25 + 22 * len(lines)
    
    # Draw rounded panel background with transparency
    # Create overlay for just the panel area
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+panel_w, y+panel_h), bg_color, -1)
    # Add subtle border
    cv2.rectangle(overlay, (x, y), (x+panel_w, y+panel_h), NEON_CYAN, 1)
    # Blend only the panel area, not the whole frame
    panel_region = img[y:y+panel_h, x:x+panel_w]
    overlay_region = overlay[y:y+panel_h, x:x+panel_w]
    blended = cv2.addWeighted(panel_region, 1.0 - alpha, overlay_region, alpha, 0)
    img[y:y+panel_h, x:x+panel_w] = blended
    
    # Draw text lines with better spacing
    for i, line in enumerate(lines):
        text_y = y + 22 + i * 22
        # Shadow
        cv2.putText(img, line, (x+10, text_y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        # Main text
        cv2.putText(img, line, (x+10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEON_CYAN, 1, cv2.LINE_AA)
    
    return img


def save_screenshot(img: np.ndarray, out_dir: Optional[str] = None) -> str:
    """Save screenshot with timestamp.
    
    Args:
        img: Image to save
        out_dir: Output directory (uses config if None)
        
    Returns:
        Path to saved file
    """
    if out_dir is None:
        out_dir = _config.get('paths.screenshots', 'screenshots')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    fname = os.path.join(out_dir, f'screenshot_{timestamp}.png')
    cv2.imwrite(fname, img)
    logger.info(f"Screenshot saved: {fname}")
    return fname


if __name__ == "__main__":
    # quick visual demo when run as script
    cap = cv2.VideoCapture(0)
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h,w = frame.shape[:2]
        # demo rounded rect in center
        rect = (w//4, h//4, w//2, h//2)
        draw_rounded_rect(frame, rect, NEON_CYAN, thickness=2, glow=True)
        y = int((time.time()-start)*150) % h
        draw_scanline(frame, y)
        draw_glitch_text(frame, 'AI FACE PERSONA', pos=(30,50))
        draw_fps(frame, 30)
        cv2.imshow('hud demo', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
