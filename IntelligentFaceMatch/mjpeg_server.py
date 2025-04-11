import socket
import threading
import logging
import time
import cv2
from typing import Dict, List, Optional, Tuple

class MJPEGServer:
    """
    Simple MJPEG streaming server that directly serves camera frames
    This is separate from the FastAPI endpoints to handle direct camera streaming
    """
    
    def __init__(self, camera_manager, stream_manager, host="0.0.0.0", port=8001):
        """
        Initialize the MJPEG server
        
        Args:
            camera_manager: CameraManager instance
            stream_manager: StreamManager instance
            host: Server host
            port: Server port
        """
        self.camera_manager = camera_manager
        self.stream_manager = stream_manager
        self.host = host
        self.port = port
        
        # Server state
        self.server_socket = None
        self.running = False
        self.thread = None
        
        # Client tracking
        self.clients = {}
        self.client_lock = threading.RLock()
        
        logging.info(f"MJPEG Server initialized on {host}:{port}")
    
    def start(self):
        """Start the MJPEG server"""
        if self.running:
            return
        
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            # Start server thread
            self.running = True
            self.thread = threading.Thread(target=self._accept_clients, daemon=True)
            self.thread.start()
            
            # Start client handler thread
            threading.Thread(target=self._process_cameras, daemon=True).start()
            
            logging.info(f"MJPEG Server started on {self.host}:{self.port}")
            return True
        
        except Exception as e:
            logging.error(f"Error starting MJPEG server: {e}")
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            return False
    
    def stop(self):
        """Stop the MJPEG server"""
        self.running = False
        
        # Close all client connections
        with self.client_lock:
            for client_id, client in self.clients.items():
                try:
                    client['socket'].close()
                except Exception:
                    pass
            self.clients.clear()
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        
        # Wait for thread to end
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        
        logging.info("MJPEG Server stopped")
    
    def _accept_clients(self):
        """Accept client connections"""
        logging.info("Starting client acceptance loop")
        
        while self.running:
            try:
                # Accept new client
                client_socket, client_addr = self.server_socket.accept()
                client_id = f"{client_addr[0]}:{client_addr[1]}"
                
                logging.info(f"New client connected: {client_id}")
                
                # Set socket options
                client_socket.settimeout(5.0)
                
                # Read HTTP request
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = client_socket.recv(1024)
                    if not chunk:
                        break
                    request += chunk
                
                if not request:
                    client_socket.close()
                    continue
                
                # Parse request to get camera ID
                request_text = request.decode('utf-8', errors='ignore')
                camera_id = self._parse_camera_id(request_text)
                
                if not camera_id:
                    # Send 404 response
                    client_socket.sendall(b"HTTP/1.1 404 Not Found\r\n\r\n")
                    client_socket.close()
                    continue
                
                # Check if camera exists
                camera = self.camera_manager.get_camera(camera_id)
                if not camera:
                    # Send 404 response
                    client_socket.sendall(b"HTTP/1.1 404 Not Found\r\n\r\n")
                    client_socket.close()
                    continue
                
                # Send HTTP header
                client_socket.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                    b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                    b"Pragma: no-cache\r\n"
                    b"Expires: 0\r\n\r\n"
                )
                
                # Add client to list
                with self.client_lock:
                    self.clients[client_id] = {
                        'socket': client_socket,
                        'camera_id': camera_id,
                        'address': client_addr,
                        'last_frame_time': time.time(),
                        'frame_count': 0
                    }
            
            except socket.timeout:
                # Timeout is normal, just continue
                continue
            except Exception as e:
                if self.running:
                    logging.error(f"Error accepting client: {e}")
                time.sleep(0.1)
    
    def _parse_camera_id(self, request_text):
        """Parse camera ID from HTTP request"""
        # Simple parsing, just look for /camera/{id} in the first line
        lines = request_text.split('\r\n')
        if not lines:
            return None
        
        first_line = lines[0]
        parts = first_line.split()
        
        if len(parts) < 2:
            return None
        
        path = parts[1]
        
        # Check if it's a camera request
        if path.startswith('/camera/'):
            return path[8:]
        
        return None
    
    def _process_cameras(self):
        """Process camera frames and send to clients"""
        while self.running:
            try:
                # Get active cameras
                camera_ids = set()
                with self.client_lock:
                    for client in self.clients.values():
                        camera_ids.add(client['camera_id'])
                
                # Process each camera
                for camera_id in camera_ids:
                    camera = self.camera_manager.get_camera(camera_id)
                    if not camera:
                        continue
                    
                    # Get frame with overlay
                    frame = camera.get_frame_with_overlay()
                    if frame is None:
                        continue
                    
                    # Encode frame to JPEG
                    _, jpg_buffer = cv2.imencode(
                        '.jpg', frame, 
                        [cv2.IMWRITE_JPEG_QUALITY, 70]
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
                    
                    # Send to clients subscribed to this camera
                    with self.client_lock:
                        for client_id, client in list(self.clients.items()):
                            if client['camera_id'] != camera_id:
                                continue
                            
                            try:
                                # Send frame
                                client['socket'].sendall(mjpeg_frame)
                                client['frame_count'] += 1
                                client['last_frame_time'] = time.time()
                            except (BrokenPipeError, ConnectionResetError):
                                # Connection closed
                                self._remove_client(client_id)
                            except Exception as e:
                                logging.error(f"Error sending frame to client {client_id}: {e}")
                                self._remove_client(client_id)
                
                # Remove stale clients
                self._clean_clients()
                
                # Sleep to control frame rate
                time.sleep(0.03)  # ~30fps max
            
            except Exception as e:
                logging.error(f"Error in camera processing: {e}")
                time.sleep(0.1)
    
    def _remove_client(self, client_id):
        """Remove a client"""
        with self.client_lock:
            if client_id in self.clients:
                try:
                    self.clients[client_id]['socket'].close()
                except Exception:
                    pass
                del self.clients[client_id]
    
    def _clean_clients(self):
        """Remove stale clients"""
        now = time.time()
        with self.client_lock:
            for client_id, client in list(self.clients.items()):
                # Remove clients with no activity for 30 seconds
                if now - client['last_frame_time'] > 30.0:
                    logging.info(f"Removing stale client: {client_id}")
                    try:
                        client['socket'].close()
                    except Exception:
                        pass
                    del self.clients[client_id]
