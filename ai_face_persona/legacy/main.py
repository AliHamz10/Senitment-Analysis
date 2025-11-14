"""
main.py
AI Face Emotion & Persona Overlay

Real-time webcam app that combines MediaPipe face detection, a Hugging Face emotion
text classifier (adapter), and a neon cyberpunk HUD overlay.

Created by Ali Hamza & Zarmeena Jawad
"""
import cv2
import time
import os
import numpy as np
try:
    import winsound
except ImportError:
    winsound = None  # macOS/Linux don't have winsound

from face_detector import FaceDetector
from emotion_model import EmotionModel
import overlay_utils as ou


def play_shutter_sound():
    """Play a short futuristic sound on Windows using winsound.Beep.
    This is lightweight and avoids additional audio dependencies.
    """
    try:
        if winsound:
            # quick modern beep sequence
            winsound.Beep(1000, 80)
            winsound.Beep(1400, 60)
    except Exception:
        pass


def main():
    import traceback
    import platform
    # create components
    cap = None
    detector = None
    emotion = None
    try:
        # Try to open camera - on macOS, try multiple indices and wait a bit
        print('Initializing camera...')
        if platform.system() == 'Darwin':  # macOS
            print('macOS detected - checking camera permissions...')
            print('If prompted, please allow camera access in System Settings > Privacy & Security > Camera')
            import time
            time.sleep(1)  # Give time for permission dialog
        
        # Try camera indices 0, 1, 2
        # On macOS, use AVFoundation backend (native)
        backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else cv2.CAP_ANY
        cap = None
        for cam_idx in [0, 1, 2]:
            print(f'Trying camera index {cam_idx}...')
            test_cap = cv2.VideoCapture(cam_idx, backend)
            if test_cap.isOpened():
                # Wait a moment for camera to initialize
                time.sleep(0.3)
                # Try to read a frame multiple times to verify it actually works
                for attempt in range(3):
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        cap = test_cap
                        print(f'Camera {cam_idx} opened successfully!')
                        break
                    time.sleep(0.2)
                if cap:
                    break
                else:
                    test_cap.release()
            else:
                if test_cap is not None:
                    test_cap.release()
        
        if cap is None or not cap.isOpened():
            print('\nError: Could not open webcam.')
            print('\nTroubleshooting steps:')
            print('1. Ensure a camera is connected')
            if platform.system() == 'Darwin':
                print('2. Grant camera permission: System Settings > Privacy & Security > Camera')
                print('   Make sure Terminal (or your IDE) has camera access enabled')
                print('3. Close other apps that might be using the camera (Zoom, FaceTime, etc.)')
            else:
                print('2. Close other apps that might be using the camera')
            print('4. Try unplugging and replugging your camera')
            return
        
        # Set camera properties for better compatibility
        # Note: Continuity Camera might not respect these, but try anyway
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Warm up camera - Continuity Camera needs more time
        print('Warming up camera (Continuity Camera may need a moment)...')
        valid_frames = 0
        for i in range(15):  # Try more frames for Continuity Camera
            ret, test_frame = cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                # Check if frame has actual content (not all black)
                if test_frame.mean() > 1.0:  # Has some brightness
                    valid_frames += 1
                    if valid_frames >= 2:  # Got at least 2 good frames
                        print(f'Camera is reading valid frames: {test_frame.shape[1]}x{test_frame.shape[0]}')
                        break
            time.sleep(0.2)  # Wait longer between attempts for Continuity Camera
        
        if valid_frames < 2:
            print('Warning: Camera opened but frames appear black or empty')
            print('   Continuity Camera may need more time to initialize')
            print('   The app will continue and should work once frames start coming through...')
        
        print('Loading AI models...')
        detector = FaceDetector()
        # Use lightweight image heuristic mode by default for stability.
        emotion = EmotionModel(mode='image')
        # prefer ONNX backend for DL to avoid TensorFlow/DeepFace native issues on Windows
        emotion.dl_backend = 'onnx'
        try:
            emotion.load()
        except Exception as e:
            print('Warning: Could not load Hugging Face model. Running in fallback (neutral).', e)
        
        print('All systems ready!')
        print('Starting webcam feed...')
        print('Press ESC to exit\n')

        fps_smooth = 30.0
        last_time = time.time()
        scan_y = 0
        display_bbox = None
        # smoothing controls (editable by user keys)
        # label smoothing decay: closer to 1.0 -> more smoothing (less jitter)
        label_decay = emotion.recent_decay
        # bbox lerp factor: 0..1 (higher -> faster following, lower -> smoother)
        bbox_lerp = emotion.bbox_lerp
        # DL toggle: start disabled for stability
        use_dl = False
        # DL backend: 'deepface' or 'onnx'
        dl_backend = emotion.dl_backend

        frame_count = 0
        black_frame_warnings = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Warning: frame read failed, exiting loop')
                break
            
            # Check if frame is valid (not None and not empty)
            if frame is None or frame.size == 0:
                if frame_count < 10:  # Only warn for first few frames
                    print('Warning: received empty frame, skipping...')
                continue
            
            # Check if frame is all black (Continuity Camera might start with black frames)
            frame_brightness = frame.mean()
            if frame_count < 30:  # Check first 30 frames
                if frame_brightness < 1.0:  # Frame is essentially black
                    black_frame_warnings += 1
                    if black_frame_warnings <= 3:  # Only warn 3 times
                        print(f'Frame {frame_count}: black frame detected (brightness={frame_brightness:.2f})')
                        print('  Continuity Camera may still be initializing...')
                elif black_frame_warnings > 0 and frame_brightness > 5.0:
                    print(f'Frame {frame_count}: Camera feed is now active! (brightness={frame_brightness:.2f})')
                    black_frame_warnings = 0  # Reset counter
                frame_count += 1
            
            # Skip processing if frame is still black after warm-up period
            if frame_count > 30 and frame_brightness < 1.0:
                # Still getting black frames after warm-up - skip this frame
                continue

            h, w = frame.shape[:2]
            bbox, lms = detector.detect(frame)

            # smooth bbox movement for HUD (interpolate previous displayed bbox)
            if bbox:
                if display_bbox is None:
                    display_bbox = bbox
                else:
                    x0,y0,w0,h0 = display_bbox
                    x1,y1,w1,h1 = bbox
                    lerp = bbox_lerp
                    nx = int(x0 + (x1 - x0) * lerp)
                    ny = int(y0 + (y1 - y0) * lerp)
                    nw = int(w0 + (w1 - w0) * lerp)
                    nh = int(h0 + (h1 - h0) * lerp)
                    display_bbox = (nx, ny, nw, nh)
            else:
                display_bbox = None

            label, conf, persona, alpha = ('neutral', 0.0, 'Calm Sentinel', 1.0)
            if bbox and emotion and use_dl:
                # if DL requested, try using DL predictor directly (requires DeepFace or ONNX)
                try:
                    label, conf, persona, alpha = emotion.predict_dl(frame, bbox)
                except Exception:
                    # fallback to landmarks
                    if lms:
                        label, conf, persona, alpha = emotion.predict(lms, (h, w))
            elif lms:
                label, conf, persona, alpha = emotion.predict(lms, (h, w))

            # overlay drawing
            if display_bbox:
                x,y,ww,hh = display_bbox
                # draw glow rounded rect (use cyan accent) using smoothed bbox
                frame = ou.draw_rounded_rect(frame, display_bbox, ou.NEON_CYAN, thickness=2, radius=22, glow=True)
                # draw emotion label with fade alpha using smoothed bbox
                ou.draw_emotion_label(frame, label, conf, persona, display_bbox, alpha)

            # animated scanline
            scan_y = (scan_y + int( (time.time()-last_time) * 180 )) % h
            ou.draw_scanline(frame, scan_y, color=ou.NEON_CYAN, thickness=2)

            # glitch header
            ou.draw_glitch_text(frame, 'AI FACE EMOTION & PERSONA OVERLAY', pos=(20, 40))

            # FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - last_time))
            last_time = now
            fps_smooth = fps_smooth * 0.85 + fps * 0.15
            ou.draw_fps(frame, fps_smooth)

            cv2.imshow('AI Face Persona', frame)

            key = cv2.waitKey(1) & 0xFF
            # interactive controls
            if key == ord('=') or key == ord('+'):
                # increase smoothing (more inertia)
                label_decay = min(0.98, label_decay + 0.03)
                emotion.recent_decay = label_decay
            elif key == ord('-'):
                label_decay = max(0.5, label_decay - 0.03)
                emotion.recent_decay = label_decay
            elif key == ord(']'):
                bbox_lerp = min(0.9, bbox_lerp + 0.05)
                emotion.bbox_lerp = bbox_lerp
            elif key == ord('['):
                bbox_lerp = max(0.02, bbox_lerp - 0.05)
                emotion.bbox_lerp = bbox_lerp
            elif key == ord('d'):
                use_dl = not use_dl
            elif key == ord('m'):
                dl_backend = 'onnx' if dl_backend == 'deepface' else 'deepface'
                emotion.dl_backend = dl_backend
            # show status panel
            status_lines = [f'Smoothing: {label_decay:.2f}  BBox lerp: {bbox_lerp:.2f}', f'DL: {"ON" if use_dl else "OFF"}  Backend: {dl_backend}', 'Keys: +/- smoothing, [/] bbox, d toggle DL, m switch backend']
            ou.draw_status_panel(frame, status_lines, pos=(10,60))
            if key == ord('s') or key == ord('S'):
                # save screenshot
                fn = ou.save_screenshot(frame)
                print('Saved', fn)
                play_shutter_sound()
            if key == 27:
                break

    except Exception:
        traceback.print_exc()
        print('Fatal error in main loop')
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
