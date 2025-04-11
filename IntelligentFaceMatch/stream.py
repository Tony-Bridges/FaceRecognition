import cv2
import threading
import time
import io
import logging
from typing import Dict, Optional

class MJPEGStreamer:
    """
    MJPEG streaming server for live camera feeds
    Handles compression and streaming of frames
    """
    
    def __init__(self):
        # Dictionary to store client connections
        self.clients = {}
        self.clients_lock = threading.RLock()
        
        # Frame parameters
        self.jpeg_quality = 70
        self.max_fps = 15
        self.min_frame_time = 1.0 / self.max_fps
        
        logging.info("MJPEG Streamer initialized")
    
    def add_client(self, client_id):
        """Add a new streaming client"""
        with self.clients_lock:
            if client_id in self.clients:
                return False
            
            self.clients[client_id] = {
                'queue': [],
                'lock': threading.Lock(),
                'last_frame_time': 0,
                'connected': True
            }
            
            logging.info(f"Added streaming client: {client_id}")
            return True
    
    def remove_client(self, client_id):
        """Remove a streaming client"""
        with self.clients_lock:
            if client_id in self.clients:
                self.clients[client_id]['connected'] = False
                del self.clients[client_id]
                logging.info(f"Removed streaming client: {client_id}")
                return True
            return False
    
    def get_client_ids(self):
        """Get all client IDs"""
        with self.clients_lock:
            return list(self.clients.keys())
    
    def send_frame(self, client_id, frame):
        """Send a frame to a specific client"""
        with self.clients_lock:
            client = self.clients.get(client_id)
            if not client or not client['connected']:
                return False
            
            # Check frame rate limiting
            now = time.time()
            if now - client['last_frame_time'] < self.min_frame_time:
                return False
            
            client['last_frame_time'] = now
            
            # Compress the frame to JPEG
            try:
                with client['lock']:
                    # Encode frame to JPEG
                    _, jpg_buffer = cv2.imencode(
                        '.jpg', frame, 
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    )
                    
                    # Convert to bytes
                    jpg_bytes = jpg_buffer.tobytes()
                    
                    # Create MJPEG frame
                    mjpeg_frame = (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(jpg_bytes)).encode() + b'\r\n'
                        b'\r\n' + jpg_bytes + b'\r\n'
                    )
                    
                    # Add to client's queue
                    client['queue'].append(mjpeg_frame)
                    
                    # Limit queue size
                    if len(client['queue']) > 3:
                        client['queue'] = client['queue'][-3:]
                    
                    return True
            
            except Exception as e:
                logging.error(f"Error sending frame to client {client_id}: {e}")
                return False
    
    def get_client_frame(self, client_id):
        """Get the next frame for a client (non-blocking)"""
        with self.clients_lock:
            client = self.clients.get(client_id)
            if not client or not client['connected']:
                return None
            
            with client['lock']:
                if client['queue']:
                    return client['queue'].pop(0)
                return None

class StreamManager:
    """
    Manages video streams from multiple cameras
    """
    
    def __init__(self):
        self.streamer = MJPEGStreamer()
        self.camera_clients = {}  # Maps camera_id to set of client_ids
        self.client_cameras = {}  # Maps client_id to camera_id
        self.lock = threading.RLock()
        
        # Stream processing thread
        self.running = False
        self.thread = None
        
        logging.info("Stream Manager initialized")
    
    def start(self):
        """Start the stream manager"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._process_streams, daemon=True)
        self.thread.start()
        
        logging.info("Stream Manager started")
    
    def stop(self):
        """Stop the stream manager"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        
        logging.info("Stream Manager stopped")
    
    def subscribe_client(self, client_id, camera_id):
        """Subscribe a client to a camera stream"""
        with self.lock:
            # Add client to streamer
            self.streamer.add_client(client_id)
            
            # Update mappings
            if camera_id not in self.camera_clients:
                self.camera_clients[camera_id] = set()
            
            self.camera_clients[camera_id].add(client_id)
            self.client_cameras[client_id] = camera_id
            
            logging.info(f"Client {client_id} subscribed to camera {camera_id}")
            return True
    
    def unsubscribe_client(self, client_id):
        """Unsubscribe a client from all streams"""
        with self.lock:
            # Remove from streamer
            self.streamer.remove_client(client_id)
            
            # Update mappings
            camera_id = self.client_cameras.get(client_id)
            if camera_id:
                if camera_id in self.camera_clients:
                    self.camera_clients[camera_id].discard(client_id)
                    if not self.camera_clients[camera_id]:
                        del self.camera_clients[camera_id]
                
                del self.client_cameras[client_id]
            
            logging.info(f"Client {client_id} unsubscribed from streams")
            return True
    
    def get_client_frame(self, client_id):
        """Get the next frame for a client"""
        return self.streamer.get_client_frame(client_id)
    
    def process_camera_frame(self, camera_id, frame):
        """Process a frame from a camera and send to subscribed clients"""
        with self.lock:
            # Check if camera has clients
            clients = self.camera_clients.get(camera_id, set())
            if not clients:
                return 0
            
            # Send frame to each client
            for client_id in list(clients):
                self.streamer.send_frame(client_id, frame)
            
            return len(clients)
    
    def _process_streams(self):
        """Background thread for processing streams"""
        while self.running:
            # This would be where we'd handle any periodic tasks
            # For now, just sleep to prevent CPU hogging
            time.sleep(0.1)
