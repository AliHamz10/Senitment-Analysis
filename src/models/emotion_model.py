"""
Emotion Model Module
Loads Hugging Face emotion classifier and provides landmark-based emotion detection.
Includes persona mapping and smooth fade animation state manager.

Created by Ali Hamza & Zarmeena Jawad
"""
import time
from typing import List, Tuple, Optional
import numpy as np
import threading
from collections import deque, defaultdict
import os
import cv2
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)

pipeline = None
DeepFace = None
onnxruntime = None

try:
    import onnxruntime as _ort
    onnxruntime = _ort
except Exception:
    onnxruntime = None


class EmotionModel:
    """EmotionModel wraps a HF text-classification pipeline for go-emotions.

    It converts simple numeric facial metrics into a textual description (e.g. "smiling, eyes open")
    and asks the text classifier to predict. This is a pragmatic adapter so we can use the
    requested model for visual emotion via heuristics.
    """

    def __init__(self, model_name: Optional[str] = None, mode: Optional[str] = None, config: Optional[Config] = None):
        """Create EmotionModel.

        Args:
            model_name: Name of HF model to use if mode=='text' (uses config if None)
            mode: 'text', 'image', 'hybrid', or 'dl' (uses config if None)
            config: Configuration instance (uses singleton if None)
        """
        self.config = config or Config()
        self.model_name = model_name or self.config.get('model.model_name', "joeddav/distilbert-base-uncased-go-emotions")
        self.mode = mode or self.config.get('model.mode', 'image')
        
        # Load persona map from config
        persona_config = self.config.get('persona', {})
        self.PERSONA_MAP = {
            'joy': persona_config.get('joy', 'AI Dreamer'),
            'surprise': persona_config.get('surprise', 'Curious Synth'),
            'anger': persona_config.get('anger', 'Chrome Rebel'),
            'sadness': persona_config.get('sadness', 'Neon Loner'),
            'confused': persona_config.get('confused', 'Quantum Puzzler'),
            'happy': persona_config.get('happy', 'Sunset Coder'),
            'excited': persona_config.get('excited', 'Pulse Rider'),
            'fear': persona_config.get('fear', 'Circuit Warden'),
            'disgust': persona_config.get('disgust', 'Acid Critic'),
            'neutral': persona_config.get('neutral', 'Calm Sentinel')
        }
        self.classifier = None
        self._load_lock = threading.Lock()
        self.last_label = None
        self.display_label = None
        self.display_alpha = 1.0
        self.last_change_time = time.time()
        # smoothing history for labels: recent predicted (label, conf)
        self.recent = deque(maxlen=8)
        # decay factor for recency weighting (0.0..1.0). Smaller -> more weight to recent
        self.recent_decay = self.config.get('model.recent_decay', 0.85)
        # bbox smoothing lerp default (can be tuned from main)
        self.bbox_lerp = self.config.get('model.bbox_lerp', 0.22)
        # DL backend selection ('deepface' or 'onnx')
        self.dl_backend = self.config.get('model.dl_backend', 'onnx')
        # cached ONNX session
        self._onnx_sess = None
        # URL to a small ONNX FER+ model (official ONNX models repo). Will be downloaded on demand.
        # Use raw.githubusercontent.com to avoid GitHub redirect issues
        self._onnx_model_url = 'https://raw.githubusercontent.com/onnx/models/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx'

    def load(self):
        # Lazy import transformers.pipeline to avoid heavy imports at module import time
        global pipeline
        try:
            if pipeline is None:
                from transformers import pipeline as _pipeline
                pipeline = _pipeline
        except Exception:
            raise RuntimeError("transformers library not available. Install requirements.txt")

        with self._load_lock:
            if self.classifier is None:
                # return_all_scores so we can pick top label and confidence
                self.classifier = pipeline("text-classification", model=self.model_name, return_all_scores=True)

    def _landmarks_to_text(self, landmarks: List[Tuple[int,int]], image_shape: Tuple[int,int]) -> str:
        """Simple heuristic to describe face as text for the text classifier.

        Uses mouth corners, eye openness and eyebrow slope heuristics to produce a short sentence.
        """
        if not landmarks:
            return "neutral face"

        h, w = image_shape
        # indices from MediaPipe face mesh for lips and eyes (approximate)
        left_mouth = landmarks[61] if len(landmarks) > 61 else landmarks[-1]
        right_mouth = landmarks[291] if len(landmarks) > 291 else landmarks[-1]
        top_lip = landmarks[13] if len(landmarks) > 13 else landmarks[-1]
        bottom_lip = landmarks[14] if len(landmarks) > 14 else landmarks[-1]

        left_eye_top = landmarks[159] if len(landmarks) > 159 else landmarks[-1]
        left_eye_bottom = landmarks[145] if len(landmarks) > 145 else landmarks[-1]
        right_eye_top = landmarks[386] if len(landmarks) > 386 else landmarks[-1]
        right_eye_bottom = landmarks[374] if len(landmarks) > 374 else landmarks[-1]

        mouth_width = np.hypot(right_mouth[0]-left_mouth[0], right_mouth[1]-left_mouth[1])
        mouth_open = np.hypot(top_lip[0]-bottom_lip[0], top_lip[1]-bottom_lip[1])

        eye_left_h = abs(left_eye_top[1] - left_eye_bottom[1])
        eye_right_h = abs(right_eye_top[1] - right_eye_bottom[1])
        eye_avg = (eye_left_h + eye_right_h) / 2.0

        # normalize by face height approx
        face_h = h
        smile_score = mouth_width / (face_h * 0.3 + 1e-6)
        open_score = mouth_open / (face_h * 0.05 + 1e-6)
        eye_open_score = eye_avg / (face_h * 0.03 + 1e-6)

        parts = []
        if smile_score > 0.28 and open_score < 0.06:
            parts.append("smiling")
        elif open_score > 0.06:
            parts.append("mouth open")
        else:
            parts.append("neutral mouth")

        if eye_open_score > 1.2:
            parts.append("eyes wide")
        elif eye_open_score < 0.6:
            parts.append("eyes squint")
        else:
            parts.append("eyes normal")

        desc = ", ".join(parts)
        return desc

    def _predict_from_landmarks(self, landmarks: List[Tuple[int,int]], image_shape: Tuple[int,int]):
        """Improved image-based heuristic classifier with better feature extraction.
        
        Computes:
        - Eye Aspect Ratio (EAR): More accurate calculation using 6 points per eye
        - Mouth Aspect Ratio (MAR): Vertical/horizontal mouth opening
        - Eyebrow position: Vertical position relative to eyes
        - Facial symmetry: Left-right asymmetry
        
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        h, w = image_shape
        if not landmarks or len(landmarks) < 468:
            return 'neutral', 0.0

        def dist(a, b):
            return np.hypot(a[0]-b[0], a[1]-b[1])
        
        def safe_landmark(idx, fallback_idx=-1):
            """Safely get landmark by index."""
            if len(landmarks) > idx:
                return landmarks[idx]
            return landmarks[fallback_idx] if fallback_idx >= 0 else landmarks[0]

        # MediaPipe Face Mesh landmark indices (468 total)
        # Left eye landmarks (6 points for better EAR)
        left_eye_outer = safe_landmark(33)   # Left eye outer corner
        left_eye_inner = safe_landmark(133)  # Left eye inner corner
        left_eye_top = safe_landmark(159)    # Left eye top
        left_eye_bottom = safe_landmark(145) # Left eye bottom
        left_eye_left = safe_landmark(33)
        left_eye_right = safe_landmark(133)
        
        # Right eye landmarks
        right_eye_outer = safe_landmark(263) # Right eye outer corner
        right_eye_inner = safe_landmark(362)  # Right eye inner corner
        right_eye_top = safe_landmark(386)    # Right eye top
        right_eye_bottom = safe_landmark(374) # Right eye bottom
        right_eye_left = safe_landmark(362)
        right_eye_right = safe_landmark(263)
        
        # Mouth landmarks
        mouth_left = safe_landmark(61)   # Left mouth corner
        mouth_right = safe_landmark(291) # Right mouth corner
        mouth_top = safe_landmark(13)    # Top lip center
        mouth_bottom = safe_landmark(14) # Bottom lip center
        mouth_center_top = safe_landmark(12)
        mouth_center_bottom = safe_landmark(15)
        
        # Eyebrow landmarks (for eyebrow position)
        left_eyebrow = safe_landmark(107)  # Left eyebrow inner
        right_eyebrow = safe_landmark(336) # Right eyebrow inner
        
        # Nose tip (for face reference)
        nose_tip = safe_landmark(4)
        
        # Calculate Eye Aspect Ratio (EAR) - improved formula
        # EAR = (vertical1 + vertical2) / (2 * horizontal)
        left_ear_vertical1 = dist(left_eye_top, left_eye_bottom)
        left_ear_vertical2 = dist(left_eye_top, left_eye_bottom)  # Can use different points
        left_ear_horizontal = dist(left_eye_outer, left_eye_inner)
        left_ear = (left_ear_vertical1 + left_ear_vertical2) / (2.0 * max(left_ear_horizontal, 1e-6))
        
        right_ear_vertical1 = dist(right_eye_top, right_eye_bottom)
        right_ear_vertical2 = dist(right_eye_top, right_eye_bottom)
        right_ear_horizontal = dist(right_eye_outer, right_eye_inner)
        right_ear = (right_ear_vertical1 + right_ear_vertical2) / (2.0 * max(right_ear_horizontal, 1e-6))
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Calculate Mouth Aspect Ratio (MAR)
        mouth_width = dist(mouth_left, mouth_right)
        mouth_height = dist(mouth_top, mouth_bottom)
        mar = mouth_height / max(mouth_width, 1e-6)
        
        # Calculate eyebrow position (relative to eyes)
        left_eyebrow_to_eye = abs(left_eyebrow[1] - left_eye_top[1])
        right_eyebrow_to_eye = abs(right_eyebrow[1] - right_eye_top[1])
        avg_eyebrow_height = (left_eyebrow_to_eye + right_eyebrow_to_eye) / 2.0
        
        # Calculate face dimensions for normalization
        face_width = dist(mouth_left, mouth_right) * 2.5  # Approximate face width
        face_height = dist(nose_tip, mouth_bottom) * 3.0   # Approximate face height
        face_size = max(face_width, face_height, h * 0.3)  # Use larger dimension
        
        # Normalize features
        norm_ear = avg_ear
        norm_mar = mar
        norm_eyebrow = avg_eyebrow_height / max(face_size, 1e-6)
        norm_mouth_width = mouth_width / max(face_size, 1e-6)
        norm_mouth_height = mouth_height / max(face_size, 1e-6)
        
        # Calculate asymmetry (for confused/surprise detection)
        eye_asymmetry = abs(left_ear - right_ear)
        eyebrow_asymmetry = abs(left_eyebrow_to_eye - right_eyebrow_to_eye) / max(face_size, 1e-6)
        
        # Compute emotion scores with improved thresholds
        # Calculate mouth corners for smile detection
        mouth_corner_left = safe_landmark(61)  # Left mouth corner
        mouth_corner_right = safe_landmark(291)  # Right mouth corner
        mouth_center_top = safe_landmark(13)  # Upper lip center
        mouth_center_bottom = safe_landmark(14)  # Lower lip center
        
        # Calculate if mouth corners are raised (smile indicator)
        mouth_corner_y_avg = (mouth_corner_left[1] + mouth_corner_right[1]) / 2.0
        mouth_center_y = (mouth_center_top[1] + mouth_center_bottom[1]) / 2.0
        mouth_upturned = (mouth_center_y - mouth_corner_y_avg) / max(face_size, 1e-6)  # Positive = upturned
        
        # Smile ratio: wider mouth relative to height indicates smile
        smile_ratio = norm_mouth_width / max(norm_mouth_height + 0.01, 0.01)
        
        # Happy: upturned mouth corners, wide smile, eyes open, eyebrows normal
        # Require actual smile indicators, not just neutral face
        happy_score = 0.0
        if mouth_upturned > 0.02 and smile_ratio > 2.5 and norm_ear > 0.15:
            happy_score = (mouth_upturned * 10.0) + (smile_ratio - 2.5) * 2.0 + (norm_ear - 0.15) * 3.0
        
        # Surprise: wide eyes, open mouth, raised eyebrows
        surprise_score = max(0.0, (norm_ear - 0.22) * 8.0) + max(0.0, (norm_mar - 0.18) * 5.0) + max(0.0, (0.06 - norm_eyebrow) * 4.0)
        
        # Anger: squinted eyes, tight mouth, lowered eyebrows, furrowed brow
        anger_score = max(0.0, (0.18 - norm_ear) * 10.0) + max(0.0, (0.12 - norm_mar) * 3.0) + max(0.0, (norm_eyebrow - 0.06) * 3.0)
        
        # Sadness: downturned mouth, slightly closed eyes, lowered eyebrows
        mouth_downturned = -mouth_upturned  # Negative upturned = downturned
        sadness_score = 0.0
        if mouth_downturned > 0.01 and norm_ear < 0.20:
            sadness_score = (mouth_downturned * 12.0) + (0.20 - norm_ear) * 4.0 + (norm_eyebrow > 0.05) * 2.0
        
        # Excited: very open mouth, wide eyes, raised eyebrows
        excited_score = 0.0
        if norm_mar > 0.22 and norm_ear > 0.22:
            excited_score = (norm_mar - 0.22) * 8.0 + (norm_ear - 0.22) * 5.0 + (norm_eyebrow < 0.05) * 2.0
        
        # Confused: asymmetric features, neutral mouth, furrowed brow
        confused_score = (eye_asymmetry > 0.06) * 3.0 + (eyebrow_asymmetry > 0.025) * 2.0 + (0.1 < norm_mar < 0.16) * 1.5
        
        # Fear: wide eyes, slightly open mouth, raised eyebrows
        fear_score = 0.0
        if norm_ear > 0.24 and 0.12 < norm_mar < 0.20:
            fear_score = (norm_ear - 0.24) * 6.0 + (norm_eyebrow < 0.05) * 3.0
        
        # Disgust: nose wrinkle (approximated by mouth position relative to nose), tight mouth
        nose_to_mouth = abs(nose_tip[1] - mouth_top[1]) / max(face_size, 1e-6)
        disgust_score = 0.0
        if nose_to_mouth < 0.14 and norm_mar < 0.13:
            disgust_score = (0.14 - nose_to_mouth) * 5.0 + (0.13 - norm_mar) * 3.0
        
        # Neutral: balanced features, moderate values
        neutral_score = 0.0
        if 0.15 < norm_ear < 0.22 and 0.08 < norm_mar < 0.15 and 0.03 < norm_eyebrow < 0.07:
            # Calculate how "neutral" the face is
            ear_neutral = 1.0 - abs(norm_ear - 0.185) / 0.035  # Closer to 0.185 is more neutral
            mar_neutral = 1.0 - abs(norm_mar - 0.115) / 0.035  # Closer to 0.115 is more neutral
            eyebrow_neutral = 1.0 - abs(norm_eyebrow - 0.05) / 0.02
            neutral_score = (ear_neutral + mar_neutral + eyebrow_neutral) / 3.0 * 2.5
        
        # Determine emotion with confidence
        scores = {
            'happy': max(0.0, happy_score),
            'surprise': max(0.0, surprise_score),
            'anger': max(0.0, anger_score),
            'sadness': max(0.0, sadness_score),
            'excited': max(0.0, excited_score),
            'confused': max(0.0, confused_score),
            'fear': max(0.0, fear_score),
            'disgust': max(0.0, disgust_score),
            'neutral': max(0.0, neutral_score)
        }
        
        # Find best emotion
        best_emotion = max(scores.items(), key=lambda x: x[1])
        label = best_emotion[0]
        raw_conf = best_emotion[1]
        
        # Normalize confidence to 0-1 range with better mapping
        if raw_conf > 0:
            # Use sigmoid-like function for confidence
            conf = min(0.95, 0.25 + (raw_conf / (1.0 + raw_conf * 0.3)) * 0.70)
        else:
            conf = 0.3  # Minimum confidence for neutral
        
        # Apply confidence thresholds per emotion (stricter thresholds)
        emotion_thresholds = {
            'happy': 0.5,      # Require stronger smile indicators
            'surprise': 0.45,
            'anger': 0.4,
            'sadness': 0.45,
            'excited': 0.5,
            'confused': 0.35,
            'fear': 0.4,
            'disgust': 0.35,
            'neutral': 0.25    # Lower threshold for neutral (default fallback)
        }
        
        if conf < emotion_thresholds.get(label, 0.3):
            label = 'neutral'
            conf = 0.4
        
        return label, float(max(0.0, min(1.0, conf)))

    def _predict_from_deepface(self, frame, bbox: Tuple[int,int,int,int]):
        """Use DeepFace to analyze the face region and return (label, confidence).

        This method requires DeepFace to be installed. It will crop the face bbox
        from `frame` (BGR) and call DeepFace.analyze. If it fails, raises or returns None.
        """
        # import DeepFace lazily to avoid heavy imports/crashes on module import
        global DeepFace
        if DeepFace is None:
            try:
                from deepface import DeepFace as _DeepFace
                DeepFace = _DeepFace
            except Exception:
                raise RuntimeError('DeepFace not available')

        x,y,w,h = bbox
        # safe crop
        H,W = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x+w)
        y2 = min(H, y+h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return 'neutral', 0.0

        # DeepFace expects RGB or BGR depending on backend; it can handle BGR
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            # result may be dict or list
            if isinstance(result, list) and result:
                result = result[0]
            emotions = result.get('emotion', {})
            if not emotions:
                return 'neutral', 0.0
            # pick top emotion
            top_label = max(emotions.items(), key=lambda kv: kv[1])[0]
            top_conf = emotions[top_label] / 100.0 if emotions[top_label] > 1 else emotions[top_label]
            # Map DeepFace emotion labels to our mapping names
            label_map = {
                'happy': 'joy',
                'sad': 'sadness',
                'angry': 'anger',
                'surprise': 'surprise',
                'neutral': 'neutral',
                'disgust': 'disgust',
                'fear': 'fear'
            }
            mapped = label_map.get(top_label.lower(), 'neutral')
            return mapped, float(top_conf)
        except Exception:
            return 'neutral', 0.0

    def _predict_from_onnx(self, frame, bbox: Tuple[int,int,int,int]):
        """Predict using a local ONNX model file placed in assets/emotion_model.onnx.

        The ONNX model should accept an image tensor and output emotion probabilities
        in a dict/order that this function expects. This is optional — if no model is
        present the method raises RuntimeError.
        """
        # Ensure onnxruntime is available
        if onnxruntime is None:
            raise RuntimeError('onnxruntime not available')

        assets_path = self.config.get('paths.models', 'ai_face_persona/assets')
        # Ensure path is relative to project root
        if not os.path.isabs(assets_path):
            # Try to find assets relative to current working directory or script location
            possible_paths = [
                assets_path,
                os.path.join('ai_face_persona', 'assets'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'ai_face_persona', 'assets'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    assets_path = path
                    break
        model_path = os.path.join(assets_path, 'emotion_model.onnx')
        # download model on demand if missing
        if not os.path.exists(model_path):
            try:
                self._ensure_onnx_model(model_path)
            except Exception as e:
                raise RuntimeError(f'Failed to acquire ONNX model: {e}')

        x,y,w,h = bbox
        H,W = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x+w)
        y2 = min(H, y+h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return 'neutral', 0.0

        # Many FER+ ONNX models expect grayscale 64x64 input. Prepare accordingly.
        try:
            img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        except Exception:
            img = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (64,64)).astype('float32')
        # normalize to 0..1 and expand to (1,1,64,64)
        img = (img / 255.0).astype('float32')
        img = img[None, None, :, :]

        # cache onnx session for speed
        if self._onnx_sess is None:
            self._onnx_sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        sess = self._onnx_sess
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: img})
        probs = np.asarray(out[0]).ravel()
        # softmax (some models already output probabilities)
        try:
            exp = np.exp(probs - np.max(probs))
            probs = exp / (exp.sum() + 1e-9)
        except Exception:
            pass

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        # FER+ original mapping
        fer_labels = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt']
        fer = fer_labels[idx] if idx < len(fer_labels) else 'neutral'
        # map to our canonical labels
        mapping = {
            'neutral': 'neutral',
            'happiness': 'happy',
            'surprise': 'surprise',
            'sadness': 'sadness',
            'anger': 'anger',
            'disgust': 'disgust',
            'fear': 'confused',
            'contempt': 'neutral'
        }
        lab = mapping.get(fer, 'neutral')
        return lab, conf

    def _ensure_onnx_model(self, model_path: str):
        """Download a small ONNX FER+ model into assets/ if it's not present.

        This downloads the official ONNX models repo's FER+ model. If requests is
        available it will be used; otherwise urllib is used.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = self._onnx_model_url
        print(f'Downloading ONNX emotion model from {url} ...')
        # try requests first
        try:
            import requests
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print('Downloaded ONNX model to', model_path)
            return
        except Exception:
            pass

        # fallback to urllib
        try:
            from urllib.request import urlopen
            with urlopen(url, timeout=30) as src, open(model_path, 'wb') as dst:
                dst.write(src.read())
            print('Downloaded ONNX model to', model_path)
            return
        except Exception as e:
            if os.path.exists(model_path):
                return
            raise RuntimeError(f'Could not download ONNX model: {e}')

    def predict(self, landmarks: List[Tuple[int,int]], image_shape: Tuple[int,int]):
        """Predict emotion label and confidence (0-1) from landmarks and image shape.

        Returns (label, confidence, persona)
        """
        # Decide which prediction mode to use: image heuristics, text model, deepface DL, or hybrid
        label = 'neutral'
        conf = 0.0
        if self.mode == 'image':
            label, conf = self._predict_from_landmarks(landmarks, image_shape)
        elif self.mode == 'text':
            # ensure classifier loaded
            if self.classifier is None:
                try:
                    self.load()
                except Exception:
                    label, conf = 'neutral', 0.0
                else:
                    desc = self._landmarks_to_text(landmarks, image_shape)
                    try:
                        scores_list = self.classifier(desc)
                        scores = scores_list[0] if isinstance(scores_list, list) and scores_list else scores_list
                        top = max(scores, key=lambda x: x['score'])
                        label = top['label']
                        conf = float(top['score'])
                    except Exception:
                        label, conf = 'neutral', 0.0
            else:
                desc = self._landmarks_to_text(landmarks, image_shape)
                try:
                    scores_list = self.classifier(desc)
                    scores = scores_list[0] if isinstance(scores_list, list) and scores_list else scores_list
                    top = max(scores, key=lambda x: x['score'])
                    label = top['label']
                    conf = float(top['score'])
                except Exception:
                    label, conf = 'neutral', 0.0
        elif self.mode == 'hybrid':  # hybrid: prefer image heuristics, fallback to text
            label, conf = self._predict_from_landmarks(landmarks, image_shape)
            # if low confidence, try text classifier
            if conf < 0.35:
                try:
                    if self.classifier is None:
                        self.load()
                    desc = self._landmarks_to_text(landmarks, image_shape)
                    scores_list = self.classifier(desc)
                    scores = scores_list[0] if isinstance(scores_list, list) and scores_list else scores_list
                    top = max(scores, key=lambda x: x['score'])
                    tlabel = top['label']
                    tconf = float(top['score'])
                    # prefer text if clearly stronger
                    if tconf > conf + 0.15:
                        label, conf = tlabel, tconf
                except Exception:
                    pass
        elif self.mode == 'dl':
            # deep learning mode using DeepFace. Needs DL to be installed.
            # To perform DL inference we need both the frame and bbox; the caller (main) should
            # first request a DL prediction by calling emotion.predict_dl(frame, bbox)
            # Here we keep a fallback to heuristics if landmarks available.
            if landmarks:
                label, conf = self._predict_from_landmarks(landmarks, image_shape)
            else:
                label, conf = 'neutral', 0.0
        else:
            # unknown mode - fallback to landmarks
            label, conf = self._predict_from_landmarks(landmarks, image_shape)

        # Append to recent history and compute smoothed label
        self.recent.append((label, conf))
        agg = defaultdict(float)
        weight = 1.0
        total_w = 0.0
        # weighted by recency (more recent -> higher weight)
        for lab, c in reversed(self.recent):
            agg[lab] += c * weight
            total_w += weight
            weight *= 0.85

        # pick label with highest aggregated score
        best_label = max(agg.items(), key=lambda kv: kv[1])[0]
        # compute averaged confidence for best label
        conf_smoothed = agg[best_label] / max(1e-6, total_w)
        persona = self.PERSONA_MAP.get(best_label.lower(), self.PERSONA_MAP.get('neutral'))

        # smooth fade animation state update --- trigger change time
        if best_label != self.last_label:
            self.last_change_time = time.time()
            self.last_label = best_label
            # reset alpha to 0 to start fade-in
            self.display_alpha = 0.0
            self.display_label = best_label

        # progress alpha towards 1.0
        dt = time.time() - self.last_change_time
        # 0.25s fade-in
        self.display_alpha = min(1.0, dt / 0.25)

        return best_label, float(max(0.0, min(1.0, conf_smoothed))), persona, self.display_alpha

    def predict_dl(self, frame, bbox: Tuple[int,int,int,int]):
        """Convenience method to call the DL predictor directly using frame+bbox.

        Returns (label, conf, persona, alpha)
        """
        try:
            if self.dl_backend == 'onnx':
                lab, conf = self._predict_from_onnx(frame, bbox)
            else:
                lab, conf = self._predict_from_deepface(frame, bbox)
        except Exception:
            lab, conf = 'neutral', 0.0
        persona = self.PERSONA_MAP.get(lab.lower(), self.PERSONA_MAP.get('neutral'))
        if lab != self.last_label:
            self.last_change_time = time.time()
            self.last_label = lab
            self.display_alpha = 0.0
        dt = time.time() - self.last_change_time
        self.display_alpha = min(1.0, dt / 0.25)
        return lab, conf, persona, self.display_alpha


if __name__ == "__main__":
    # basic test
    em = EmotionModel()
    try:
        em.load()
        print('Loaded model (may download first time).')
    except Exception as e:
        print('Could not load transformer model:', e)
