import os
import time
import logging
import grpc
import signal
import sys
from concurrent import futures
import threading

# Import proto-generated code (would be generated from proto file)
# import face_recognition_pb2
# import face_recognition_pb2_grpc

from face_recognition_service import FaceRecognitionServicer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class GrpcServer:
    """gRPC server for face recognition service"""
    
    def __init__(self, host='0.0.0.0', port=8000, max_workers=10):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.server = None
        self.stop_event = threading.Event()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logging.info(f"Initializing gRPC server on {host}:{port}")
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        logging.info(f"Received signal {signum}, initiating shutdown")
        self.stop()
    
    def start(self):
        """Start the gRPC server"""
        # Create server
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024)
            ]
        )
        
        # Get database URL from environment
        db_url = os.environ.get('DATABASE_URL')
        
        # Add servicer
        servicer = FaceRecognitionServicer(db_url)
        
        # In a real implementation, we would register the servicer with the generated code
        # face_recognition_pb2_grpc.add_FaceRecognitionServiceServicer_to_server(servicer, self.server)
        
        # Add insecure port
        self.server.add_insecure_port(f"{self.host}:{self.port}")
        
        # Start server
        self.server.start()
        logging.info(f"gRPC server started on {self.host}:{self.port}")
        
        # Run until stopped
        try:
            self.stop_event.wait()
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the gRPC server"""
        if self.server:
            logging.info("Stopping gRPC server")
            self.server.stop(grace=5)  # Give 5 seconds for in-progress RPCs to complete
            self.stop_event.set()
            logging.info("gRPC server stopped")

def run_server():
    """Run the gRPC server"""
    port = int(os.environ.get('GRPC_PORT', 8000))
    server = GrpcServer(port=port)
    server.start()

if __name__ == "__main__":
    run_server()
