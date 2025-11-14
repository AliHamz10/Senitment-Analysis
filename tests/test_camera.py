#!/usr/bin/env python3
"""
Simple camera test to verify camera access works on macOS

Created by Ali Hamza & Zarmeena Jawad
"""
import cv2
import sys
import platform

print("Camera Test Script")
print("=" * 50)

if platform.system() == 'Darwin':
    print("macOS detected")
    print("\nIMPORTANT: If you see a permission dialog, click 'OK' to allow camera access")
    print("   If no dialog appears, grant permission manually:")
    print("   System Settings > Privacy & Security > Camera")
    print("   Enable access for Terminal (or your IDE)\n")

print("Attempting to open camera...")
import time

# Try different backends on macOS
backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]  # AVFoundation is macOS native

cap = None
for backend in backends:
    print(f"Trying backend {backend}...")
    test_cap = cv2.VideoCapture(0, backend)
    if test_cap.isOpened():
        # Set some properties
        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Wait a moment for camera to initialize
        time.sleep(0.5)
        # Try reading multiple times
        for attempt in range(5):
            ret, frame = test_cap.read()
            if ret and frame is not None:
                cap = test_cap
                print(f"Camera opened with backend {backend}!")
                break
            time.sleep(0.2)
        if cap:
            break
        test_cap.release()

if not cap:
    print("Failed to open camera index 0")
    print("\nTrying other camera indices...")
    for i in [1, 2]:
        test_cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if test_cap.isOpened():
            time.sleep(0.5)
            for attempt in range(5):
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    print(f"Camera {i} works!")
                    cap = test_cap
                    break
                time.sleep(0.2)
            if cap:
                break
            test_cap.release()
    
    if not cap or not cap.isOpened():
        print("\nNo working camera found")
        print("\nTroubleshooting:")
        print("1. Check System Settings > Privacy & Security > Camera")
        print("2. Close other apps using the camera (Zoom, FaceTime, Photo Booth, etc.)")
        print("3. Try unplugging/replugging external cameras")
        print("4. Restart your Mac if permissions were just granted")
        sys.exit(1)

print("Camera opened successfully!")
print("Reading test frame...")

# Try reading multiple frames (Continuity Camera might need warm-up)
print("Reading frames (Continuity Camera may need a moment)...")
frames_read = 0
for i in range(10):
    ret, frame = cap.read()
    if ret and frame is not None:
        frames_read += 1
        h, w = frame.shape[:2]
        mean_val = frame.mean()
        print(f"Frame {i+1}: {w}x{h}, mean brightness: {mean_val:.2f}")
        if mean_val > 5.0:  # Not completely black
            print(f"\nValid frame captured: {w}x{h} pixels")
            print(f"  Frame brightness: {mean_val:.2f} (should be > 5 for visible image)")
            print("\nCamera is working! Displaying frame...")
            print("Press any key to close this test window...")
            
            # Resize if too large for display
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * 1280 / w)))
            
            cv2.imshow('Camera Test - Press any key to close', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
    time.sleep(0.2)

if frames_read == 0:
    print("Could not read any valid frames from camera")
    print("\nTroubleshooting for Continuity Camera:")
    print("1. Make sure iPhone is connected and Continuity Camera is active")
    print("2. Check iPhone: Settings > General > AirPlay & Handoff > Continuity Camera")
    print("3. Try disconnecting and reconnecting your iPhone")
    print("4. Make sure no other app is using the camera")
    sys.exit(1)
elif frames_read < 3:
    print(f"\nWarning: Only read {frames_read} frames - camera might be slow to initialize")

cap.release()
print("Test complete!")

