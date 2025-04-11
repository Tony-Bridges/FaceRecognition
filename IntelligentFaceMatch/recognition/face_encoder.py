import os
import numpy as np
import cv2
import logging
import threading
from typing import Tuple, List, Dict, Optional, Union

# Optional InsightFace import - falls back to OpenCV DNN if not available
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available, falling back to OpenCV DNN")

class FaceEncoder:
    """
    Face encoding using modern deep learning models
    Supports both InsightFace and OpenCV DNN backends
    """
    
    def __init__(self, model_path=None, backend="insightface", device="cpu"):
        """
        Initialize face encoder
        
        Args:
            model_path: Path to model files (optional)
            backend: 'insightface', 'arcface', or 'opencv'
            device: 'cpu' or 'cuda'
        """
        self.backend = backend
        self.device = device
        self.model_path = model_path
        self.model = None
        self.model_lock = threading.Lock()
        
        # Initialize encoder based on selected backend
        self._initialize_encoder()
        
        logging.info(f"Face encoder initialized with {backend} backend on {device}")
    
    def _initialize_encoder(self):
        """Initialize the appropriate face encoding model"""
        if self.backend == "insightface" and INSIGHTFACE_AVAILABLE:
            self._initialize_insightface()
        elif self.backend == "arcface" and INSIGHTFACE_AVAILABLE:
            self._initialize_arcface()
        else:
            self._initialize_opencv_dnn()
    
    def _initialize_insightface(self):
        """Initialize InsightFace model"""
        try:
            # Configure InsightFace
            app = FaceAnalysis(name="buffalo_l")
            ctx_id = 0 if self.device == "cuda" else -1
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.model = app
        except Exception as e:
            logging.error(f"Error initializing InsightFace: {e}")
            self._initialize_opencv_dnn()  # Fallback
    
    def _initialize_arcface(self):
        """Initialize ArcFace model from InsightFace"""
        try:
            from insightface.model_zoo import get_model
            ctx_id = 0 if self.device == "cuda" else -1
            model_path = self.model_path or os.path.expanduser('~/.insightface/models/buffalo_l/w600k_r50.onnx')
            
            if not os.path.exists(model_path):
                logging.warning(f"ArcFace model not found at: {model_path}")
                self._initialize_opencv_dnn()
                return
                
            self.model = get_model(model_path, ctx_id=ctx_id)
            self.model.prepare(ctx_id=ctx_id)
        except Exception as e:
            logging.error(f"Error initializing ArcFace: {e}")
            self._initialize_opencv_dnn()  # Fallback
    
    def _initialize_opencv_dnn(self):
        """Initialize OpenCV DNN face recognition model"""
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
            
            # Load face detector
            self.face_detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            
            # Load face recognition model (OpenFace or similar)
            # Note: This is a simplified implementation
            # In a real system, you'd use a proper face recognition model
            self.backend = "opencv"
            logging.info("Using OpenCV DNN for face detection and encoding")
        except Exception as e:
            logging.error(f"Error initializing OpenCV DNN: {e}")
            raise RuntimeError("Failed to initialize any face recognition backend")
    
    def _download_opencv_models(self, model_path):
        """Download OpenCV DNN models"""
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
    
    def detect_and_encode(self, image):
        """
        Detect faces in image and compute embeddings
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of dicts with face information
        """
        with self.model_lock:
            if self.backend == "insightface" or self.backend == "arcface":
                return self._detect_and_encode_insightface(image)
            else:
                return self._detect_and_encode_opencv(image)
    
    def _detect_and_encode_insightface(self, image):
        """Detect and encode faces using InsightFace"""
        # Convert BGR to RGB for InsightFace
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            # Get face analysis
            faces = self.model.get(rgb_image)
            
            results = []
            for face in faces:
                # Extract data
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                
                # Face landmarks (if available)
                landmarks = face.landmark if hasattr(face, 'landmark') else None
                
                results.append({
                    'rect': (int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])),
                    'embedding': embedding,
                    'landmarks': landmarks,
                    'confidence': face.det_score if hasattr(face, 'det_score') else 1.0
                })
            
            return results
        except Exception as e:
            logging.error(f"InsightFace detection error: {e}")
            return []
    
    def _detect_and_encode_opencv(self, image):
        """Detect and encode faces using OpenCV DNN"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
                
            # Get face coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure coordinates are within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Extract face ROI
            face_roi = image[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
                
            # Generate a simple embedding using HOG descriptor
            # Note: In a real system, you'd use a proper face embedding model
            try:
                # Resize face to consistent size
                face_resized = cv2.resize(face_roi, (96, 96))
                # Convert to grayscale
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                # Compute HOG descriptor as a simple embedding
                hog = cv2.HOGDescriptor((96, 96), (32, 32), (16, 16), (8, 8), 9)
                embedding = hog.compute(face_gray).flatten()
                
                # Normalize embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                results.append({
                    'rect': (x1, y1, x2-x1, y2-y1),
                    'embedding': embedding,
                    'landmarks': None,
                    'confidence': float(confidence)
                })
            except Exception as e:
                logging.error(f"Error computing face embedding: {e}")
        
        return results
    
    def encode_face(self, face_image):
        """
        Encode a pre-cropped face image
        
        Args:
            face_image: OpenCV image containing only the face
            
        Returns:
            Face embedding vector
        """
        # Resize face to expected input size
        if self.backend == "insightface" or self.backend == "arcface":
            face_resized = cv2.resize(face_image, (112, 112))
        else:
            face_resized = cv2.resize(face_image, (96, 96))
        
        # Process with the same method as detection
        results = self.detect_and_encode(face_resized)
        
        if results:
            return results[0]['embedding']
        return None
    
    def compare_faces(self, embedding1, embedding2):
        """
        Compare two face embeddings
        
        Args:
            embedding1, embedding2: Face embedding vectors
            
        Returns:
            similarity score (0-1), higher means more similar
        """
        if embedding1 is None or embedding2 is None:
            return 0
            
        # Convert to numpy arrays if needed
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)
            
        # Ensure vectors have same length
        if embedding1.shape != embedding2.shape:
            logging.error(f"Embedding shape mismatch: {embedding1.shape} vs {embedding2.shape}")
            return 0
            
        # Compute cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Convert to a 0-1 range (cosine similarity is between -1 and 1)
        similarity = (cosine_sim + 1) / 2
        
        return float(similarity)
