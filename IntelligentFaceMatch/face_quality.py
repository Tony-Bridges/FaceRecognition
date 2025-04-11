import cv2
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class FaceQualityMetrics:
    """Class for holding face quality metrics"""
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    face_size: float = 0.0
    pose_deviation: float = 0.0
    eye_openness: float = 0.0
    overall_score: float = 0.0
    
    def to_dict(self):
        """Convert metrics to dictionary"""
        return {
            'sharpness': self.sharpness,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'face_size': self.face_size,
            'pose_deviation': self.pose_deviation,
            'eye_openness': self.eye_openness,
            'overall_score': self.overall_score
        }

class FaceQualityAnalyzer:
    """Analyzer for measuring face image quality"""
    
    def __init__(self):
        logging.info("Face Quality Analyzer initialized")
    
    def analyze_face(self, image, face_rect, landmarks=None):
        """
        Analyze face quality based on various metrics
        
        Args:
            image: OpenCV image (BGR)
            face_rect: (x, y, width, height) tuple
            landmarks: Optional facial landmarks array
        
        Returns:
            FaceQualityMetrics object
        """
        metrics = FaceQualityMetrics()
        
        if image is None or face_rect is None:
            return metrics
        
        try:
            # Extract face region
            x, y, w, h = face_rect
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > image.shape[1] or y+h > image.shape[0]:
                logging.warning("Invalid face rectangle for quality analysis")
                return metrics
            
            face_img = image[y:y+h, x:x+w]
            if face_img.size == 0:
                logging.warning("Empty face image for quality analysis")
                return metrics
            
            # Convert to grayscale for certain metrics
            if len(face_img.shape) == 3:
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_img.copy()
            
            # Calculate metrics
            metrics.sharpness = self._calculate_sharpness(gray_face)
            metrics.brightness, metrics.contrast = self._calculate_brightness_contrast(gray_face)
            metrics.face_size = self._calculate_face_size(w, h, image.shape[1], image.shape[0])
            
            if landmarks is not None:
                metrics.pose_deviation = self._calculate_pose_deviation(landmarks)
                metrics.eye_openness = self._calculate_eye_openness(landmarks)
            
            # Calculate overall score (weighted average)
            weights = {
                'sharpness': 0.25,
                'brightness': 0.15,
                'contrast': 0.15,
                'face_size': 0.2,
                'pose_deviation': 0.15,
                'eye_openness': 0.1
            }
            
            overall_score = (
                weights['sharpness'] * metrics.sharpness +
                weights['brightness'] * metrics.brightness +
                weights['contrast'] * metrics.contrast +
                weights['face_size'] * metrics.face_size
            )
            
            if landmarks is not None:
                overall_score += (
                    weights['pose_deviation'] * metrics.pose_deviation +
                    weights['eye_openness'] * metrics.eye_openness
                )
            else:
                # Redistribute weights if no landmarks
                factor = 1.0 / (1.0 - weights['pose_deviation'] - weights['eye_openness'])
                overall_score *= factor
            
            metrics.overall_score = min(1.0, max(0.0, overall_score))
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error in face quality analysis: {e}")
            return metrics
    
    def _calculate_sharpness(self, gray_img):
        """Calculate sharpness using Laplacian variance"""
        if gray_img is None or gray_img.size == 0:
            return 0.0
            
        lap_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        # Normalize to 0-1 range (empirical values)
        normalized = min(1.0, max(0.0, lap_var / 1000.0))
        return normalized
    
    def _calculate_brightness_contrast(self, gray_img):
        """Calculate brightness and contrast"""
        if gray_img is None or gray_img.size == 0:
            return 0.0, 0.0
            
        # Brightness: mean pixel value normalized to 0-1
        mean_val = np.mean(gray_img) / 255.0
        
        # Ideal brightness is around 0.5 (127 in 0-255 range)
        brightness_score = 1.0 - 2.0 * abs(mean_val - 0.5)
        
        # Contrast: standard deviation normalized
        std_val = np.std(gray_img) / 128.0
        contrast_score = min(1.0, max(0.0, std_val))
        
        return brightness_score, contrast_score
    
    def _calculate_face_size(self, face_width, face_height, img_width, img_height):
        """Calculate face size relative to image size"""
        face_area = face_width * face_height
        img_area = img_width * img_height
        
        # Calculate face size as percentage of image
        size_ratio = face_area / img_area
        
        # Ideal face size is 15-50% of image
        if size_ratio < 0.05:
            # Too small
            return size_ratio * 20  # Scale up to reach 1.0 at 5%
        elif size_ratio > 0.8:
            # Too large
            return max(0.0, 1.0 - (size_ratio - 0.8) * 5)  # Scale down from 0.8
        elif 0.15 <= size_ratio <= 0.5:
            # Ideal range
            return 1.0
        else:
            # Acceptable but not ideal
            return 0.7
    
    def _calculate_pose_deviation(self, landmarks):
        """Calculate face pose deviation from frontal position"""
        # This is a simplified implementation
        # Ideally use 3D pose estimation for better accuracy
        
        if landmarks is None or len(landmarks) < 68:
            return 0.5  # Default value if landmarks unavailable
        
        try:
            # Use key facial landmarks to estimate pose
            # Left eye, right eye, nose tip, left mouth, right mouth
            left_eye = np.mean(landmarks[36:42], axis=0)
            right_eye = np.mean(landmarks[42:48], axis=0)
            nose_tip = landmarks[30]
            left_mouth = landmarks[48]
            right_mouth = landmarks[54]
            
            # Calculate horizontal symmetry
            eye_line = right_eye - left_eye
            eye_line_norm = np.linalg.norm(eye_line)
            if eye_line_norm == 0:
                return 0.5
                
            eye_line = eye_line / eye_line_norm
            
            # Project nose to eye line
            nose_to_left = nose_tip - left_eye
            proj = np.dot(nose_to_left, eye_line)
            ideal_proj = eye_line_norm / 2.0
            
            # Calculate deviation
            horizontal_dev = abs(proj - ideal_proj) / ideal_proj
            
            # Check vertical alignment
            eye_midpoint = (left_eye + right_eye) / 2.0
            mouth_midpoint = (left_mouth + right_mouth) / 2.0
            
            vertical_vec = mouth_midpoint - eye_midpoint
            vertical_vec_norm = np.linalg.norm(vertical_vec)
            if vertical_vec_norm == 0:
                return 0.5
                
            vertical_vec = vertical_vec / vertical_vec_norm
            
            # Ideal vector is straight down (0, 1)
            vertical_dev = abs(vertical_vec[0])  # How far from vertical
            
            # Combine deviations (lower is better)
            combined_dev = (horizontal_dev + vertical_dev) / 2.0
            
            # Convert to quality score (higher is better)
            pose_score = max(0.0, 1.0 - combined_dev)
            
            return pose_score
            
        except Exception:
            return 0.5  # Default value on error
    
    def _calculate_eye_openness(self, landmarks):
        """Calculate eye openness from landmarks"""
        if landmarks is None or len(landmarks) < 68:
            return 0.5  # Default value if landmarks unavailable
        
        try:
            # Calculate eye aspect ratio for both eyes
            def eye_aspect_ratio(eye_points):
                # Compute the euclidean distances between the vertical eye landmarks
                A = np.linalg.norm(eye_points[1] - eye_points[5])
                B = np.linalg.norm(eye_points[2] - eye_points[4])
                
                # Compute the euclidean distance between the horizontal eye landmarks
                C = np.linalg.norm(eye_points[0] - eye_points[3])
                
                # Compute the eye aspect ratio
                ear = (A + B) / (2.0 * C)
                return ear
            
            # Extract landmarks for left and right eyes
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            # Calculate the eye aspect ratio for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # Average the eye aspect ratio
            ear = (left_ear + right_ear) / 2.0
            
            # Normalize to 0-1 (empirical values based on typical EAR ranges)
            # Closed eyes have EAR around 0.1-0.2, open eyes 0.25-0.35
            if ear < 0.15:
                return 0.0  # Eyes closed
            elif ear > 0.35:
                return 1.0  # Eyes wide open
            else:
                # Linear scaling between 0.15 and 0.35
                return (ear - 0.15) / 0.2
                
        except Exception:
            return 0.5  # Default value on error
