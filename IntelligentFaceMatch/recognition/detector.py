import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Dict, Optional, Union

class FaceDetector:
    """
    Modern face detector using deep learning models
    Supports multiple backends: OpenCV DNN, MediaPipe, etc.
    """
    
    def __init__(self, model_path=None, backend="opencv", device="cpu"):
        """
        Initialize face detector
        
        Args:
            model_path: Path to model files (optional)
            backend: 'opencv', 'mediapipe', 'dlib'
            device: 'cpu' or 'cuda'
        """
        self.backend = backend
        self.device = device
        self.model_path = model_path
        self.detector = None
        
        # Initialize the appropriate detector
        self._initialize_detector()
        
        logging.info(f"Face detector initialized with {backend} backend on {device}")
    
    def _initialize_detector(self):
        """Initialize the appropriate face detection model"""
        if self.backend == "opencv":
            self._initialize_opencv_dnn()
        elif self.backend == "mediapipe":
            self._initialize_mediapipe()
        elif self.backend == "dlib":
            self._initialize_dlib()
        else:
            logging.warning(f"Unknown backend: {self.backend}, falling back to OpenCV")
            self.backend = "opencv"
            self._initialize_opencv_dnn()
    
    def _initialize_opencv_dnn(self):
        """Initialize OpenCV DNN face detector"""
        try:
            # Check for model files
            model_path = self.model_path or "models"
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
            
            # Model files
            prototxt = os.path.join(model_path, "deploy.prototxt")
            caffemodel = os.path.join(model_path, "res10_300x300_ssd_iter_140000.caffemodel")
            
            # Download models if not exist
            if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
                self._download_opencv_models(model_path)
            
            # Load model
            self.detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            
            # Set backend and target
            if self.device == "cuda" and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
        except Exception as e:
            logging.error(f"Error initializing OpenCV DNN detector: {e}")
            self.detector = None
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe face detector"""
        try:
            import mediapipe as mp
            
            # Initialize MediaPipe Face Detection
            mp_face_detection = mp.solutions.face_detection
            self.detector = mp_face_detection.FaceDetection(
                model_selection=1,  # 0=short range, 1=full range
                min_detection_confidence=0.5
            )
        except ImportError:
            logging.error("MediaPipe not installed, falling back to OpenCV")
            self.backend = "opencv"
            self._initialize_opencv_dnn()
        except Exception as e:
            logging.error(f"Error initializing MediaPipe detector: {e}")
            self.backend = "opencv"
            self._initialize_opencv_dnn()
    
    def _initialize_dlib(self):
        """Initialize dlib face detector"""
        try:
            import dlib
            
            # Initialize dlib detector
            self.detector = dlib.get_frontal_face_detector()
        except ImportError:
            logging.error("dlib not installed, falling back to OpenCV")
            self.backend = "opencv"
            self._initialize_opencv_dnn()
        except Exception as e:
            logging.error(f"Error initializing dlib detector: {e}")
            self.backend = "opencv"
            self._initialize_opencv_dnn()
    
    def _download_opencv_models(self, model_path):
        """Download OpenCV DNN models if they don't exist"""
        import urllib.request
        
        # URLs for the models
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        # Download files
        try:
            prototxt_path = os.path.join(model_path, "deploy.prototxt")
            caffemodel_path = os.path.join(model_path, "res10_300x300_ssd_iter_140000.caffemodel")
            
            logging.info("Downloading OpenCV face detection models...")
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
            urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
            logging.info("Model download completed")
        except Exception as e:
            logging.error(f"Error downloading models: {e}")
            raise
    
    def detect_faces(self, image, min_confidence=0.5):
        """
        Detect faces in an image
        
        Args:
            image: OpenCV image (BGR format)
            min_confidence: Minimum detection confidence (0-1)
            
        Returns:
            List of dicts with detected faces information
        """
        if self.detector is None:
            return []
            
        if self.backend == "opencv":
            return self._detect_opencv_dnn(image, min_confidence)
        elif self.backend == "mediapipe":
            return self._detect_mediapipe(image, min_confidence)
        elif self.backend == "dlib":
            return self._detect_dlib(image)
        
        return []
    
    def _detect_opencv_dnn(self, image, min_confidence):
        """Detect faces using OpenCV DNN"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < min_confidence:
                continue
                
            # Get face coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure coordinates are within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            faces.append({
                'rect': (x1, y1, x2-x1, y2-y1),
                'confidence': float(confidence),
                'landmarks': None
            })
        
        return faces
    
    def _detect_mediapipe(self, image, min_confidence):
        """Detect faces using MediaPipe"""
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Process the image
        results = self.detector.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                if detection.score[0] < min_confidence:
                    continue
                    
                # Extract bounding box
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Extract landmarks
                landmarks = []
                for landmark in detection.location_data.relative_keypoints:
                    landmarks.append((int(landmark.x * w), int(landmark.y * h)))
                
                faces.append({
                    'rect': (x1, y1, width, height),
                    'confidence': float(detection.score[0]),
                    'landmarks': landmarks
                })
        
        return faces
    
    def _detect_dlib(self, image):
        """Detect faces using dlib"""
        # Convert to RGB for dlib
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        dlib_rects = self.detector(rgb_image, 1)
        
        faces = []
        for rect in dlib_rects:
            x1 = rect.left()
            y1 = rect.top()
            x2 = rect.right()
            y2 = rect.bottom()
            
            faces.append({
                'rect': (x1, y1, x2-x1, y2-y1),
                'confidence': 1.0,  # dlib doesn't provide confidence scores
                'landmarks': None
            })
        
        return faces
    
    def extract_faces(self, image, padding=0.0):
        """
        Detect and extract face regions from image
        
        Args:
            image: OpenCV image (BGR format)
            padding: Padding factor to add around detected faces (0.0-1.0)
            
        Returns:
            List of extracted face images and their coordinates
        """
        faces = self.detect_faces(image)
        h, w = image.shape[:2]
        
        extracted = []
        for face in faces:
            x, y, width, height = face['rect']
            
            # Add padding
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            # Ensure coordinates are within image boundaries
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + width + pad_x)
            y2 = min(h, y + height + pad_y)
            
            # Extract face region
            face_img = image[y1:y2, x1:x2]
            
            if face_img.size > 0:  # Ensure valid image
                extracted.append({
                    'image': face_img,
                    'rect': (x1, y1, x2-x1, y2-y1),
                    'original_rect': face['rect'],
                    'confidence': face['confidence'],
                    'landmarks': face['landmarks']
                })
        
        return extracted
