import cv2
import numpy as np
import threading
import time
import logging
import queue
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class CameraFrame:
    """Container for camera frame data"""
    frame: np.ndarray
    timestamp: float
    camera_id: str
    frame_id: int
    width: int
    height: int
    
    def copy(self):
        """Create a deep copy of the frame"""
        return CameraFrame(
            frame=self.frame.copy(),
            timestamp=self.timestamp,
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            width=self.width,
            height=self.height
        )

class Camera:
    """
    Camera handler that supports various sources (webcam, IP camera)
    and manages frame processing and streaming
    """
    
    def __init__(self, camera_id, source=0, name=None, config=None):
        """
        Initialize camera
        
        Args:
            camera_id: Unique identifier for the camera
            source: Camera source (device ID or URL)
            name: Human-readable camera name
            config: Additional configuration parameters
        """
        self.camera_id = camera_id
        self.source = source
        self.name = name or f"Camera {camera_id}"
        self.config = config or {}
        
        # Camera state
        self.cap = None
        self.running = False
        self.thread = None
        self.lock = threading.RLock()
        self.last_frame = None
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0
        
        # Frame queue for consumers
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Frame resolution
        self.width = self.config.get('width', 640)
        self.height = self.config.get('height', 480)
        
        # Recognition results
        self.recognition_results = []
        self.recognition_lock = threading.Lock()
        
        logging.info(f"Camera {camera_id} initialized with source {source}")
    
    def start(self):
        """Start camera capture"""
        if self.running:
            return False
        
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                logging.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Get actual resolution
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Start capture thread
            self.running = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            logging.info(f"Camera {self.camera_id} started with resolution {self.width}x{self.height}")
            return True
        
        except Exception as e:
            logging.error(f"Error starting camera {self.camera_id}: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logging.info(f"Camera {self.camera_id} stopped")
    
    def get_frame(self):
        """Get the latest frame (thread-safe)"""
        with self.lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()
    
    def get_info(self):
        """Get camera information"""
        return {
            'camera_id': self.camera_id,
            'name': self.name,
            'source': self.source,
            'resolution': f"{self.width}x{self.height}",
            'fps': round(self.fps, 1),
            'running': self.running,
            'frame_count': self.frame_count
        }
    
    def update_recognition_results(self, results):
        """Update face recognition results"""
        with self.recognition_lock:
            self.recognition_results = results
    
    def get_recognition_results(self):
        """Get current recognition results"""
        with self.recognition_lock:
            return self.recognition_results.copy() if self.recognition_results else []
    
    def get_frame_with_overlay(self):
        """Get frame with recognition overlay"""
        frame = self.get_frame()
        if frame is None:
            return None
        
        # Create a copy for drawing
        img = frame.frame.copy()
        
        # Get recognition results
        results = self.get_recognition_results()
        
        # Draw results on frame
        for result in results:
            if 'rect' in result:
                rect = result['rect']
                x, y, w, h = rect
                
                # Draw rectangle
                color = (0, 255, 0) if result.get('confidence', 0) > 0.8 else (0, 165, 255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # Draw name and confidence
                face_id = result.get('face_id', 'unknown')
                conf = result.get('confidence', 0)
                text = f"{face_id} ({conf:.2f})"
                
                cv2.putText(
                    img, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        
        # Add camera info
        cv2.putText(
            img, f"FPS: {self.fps:.1f}", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        
        return img
    
    def _capture_loop(self):
        """Main capture loop (runs in separate thread)"""
        fps_update_time = time.time()
        fps_frame_count = 0
        
        while self.running:
            try:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    logging.warning(f"Failed to read frame from camera {self.camera_id}")
                    # Try to reconnect for IP cameras
                    if isinstance(self.source, str) and (self.source.startswith('rtsp://') or 
                                                         self.source.startswith('http://')):
                        self.cap.release()
                        time.sleep(1.0)
                        self.cap = cv2.VideoCapture(self.source)
                    continue
                
                # Update frame count and FPS
                self.frame_count += 1
                fps_frame_count += 1
                
                now = time.time()
                if now - fps_update_time > 1.0:
                    self.fps = fps_frame_count / (now - fps_update_time)
                    fps_update_time = now
                    fps_frame_count = 0
                
                # Create frame object
                camera_frame = CameraFrame(
                    frame=frame,
                    timestamp=now,
                    camera_id=self.camera_id,
                    frame_id=self.frame_count,
                    width=self.width,
                    height=self.height
                )
                
                # Update last frame (thread-safe)
                with self.lock:
                    self.last_frame = camera_frame
                
                # Add to queue for consumers (non-blocking)
                try:
                    self.frame_queue.put(camera_frame, block=False)
                except queue.Full:
                    # Skip frame if queue is full
                    pass
            
            except Exception as e:
                logging.error(f"Error in camera {self.camera_id} capture loop: {e}")
                time.sleep(0.1)
        
        logging.info(f"Camera {self.camera_id} capture loop ended")

class CameraManager:
    """
    Manages multiple cameras and provides access to them
    """
    
    def __init__(self):
        self.cameras = {}
        self.lock = threading.RLock()
        logging.info("Camera Manager initialized")
    
    def add_camera(self, camera_id=None, source=0, name=None, config=None, start=True):
        """
        Add a new camera
        
        Args:
            camera_id: Camera ID (generated if None)
            source: Camera source (device ID or URL)
            name: Camera name
            config: Camera configuration
            start: Whether to start the camera immediately
        
        Returns:
            Camera ID
        """
        with self.lock:
            # Generate camera ID if not provided
            if camera_id is None:
                camera_id = str(uuid.uuid4())
            
            # Check if camera already exists
            if camera_id in self.cameras:
                logging.warning(f"Camera {camera_id} already exists, updating parameters")
                camera = self.cameras[camera_id]
                
                # Update parameters
                camera.source = source
                if name:
                    camera.name = name
                if config:
                    camera.config.update(config)
                
                # Restart if running
                if camera.running and start:
                    camera.stop()
                    camera.start()
                elif start and not camera.running:
                    camera.start()
                
                return camera_id
            
            # Create new camera
            camera = Camera(camera_id, source, name, config)
            self.cameras[camera_id] = camera
            
            # Start camera if requested
            if start:
                camera.start()
            
            logging.info(f"Added camera {camera_id}")
            return camera_id
    
    def remove_camera(self, camera_id):
        """Remove a camera"""
        with self.lock:
            if camera_id in self.cameras:
                camera = self.cameras[camera_id]
                camera.stop()
                del self.cameras[camera_id]
                logging.info(f"Removed camera {camera_id}")
                return True
            return False
    
    def get_camera(self, camera_id):
        """Get a camera by ID"""
        with self.lock:
            return self.cameras.get(camera_id)
    
    def get_all_cameras(self):
        """Get all cameras"""
        with self.lock:
            return list(self.cameras.values())
    
    def get_camera_ids(self):
        """Get all camera IDs"""
        with self.lock:
            return list(self.cameras.keys())
    
    def start_camera(self, camera_id):
        """Start a camera"""
        with self.lock:
            camera = self.cameras.get(camera_id)
            if camera:
                return camera.start()
            return False
    
    def stop_camera(self, camera_id):
        """Stop a camera"""
        with self.lock:
            camera = self.cameras.get(camera_id)
            if camera:
                camera.stop()
                return True
            return False
    
    def stop_all_cameras(self):
        """Stop all cameras"""
        with self.lock:
            for camera in self.cameras.values():
                camera.stop()
    
    def get_camera_info(self, camera_id=None):
        """Get camera information"""
        with self.lock:
            if camera_id:
                camera = self.cameras.get(camera_id)
                return camera.get_info() if camera else None
            else:
                return [camera.get_info() for camera in self.cameras.values()]
