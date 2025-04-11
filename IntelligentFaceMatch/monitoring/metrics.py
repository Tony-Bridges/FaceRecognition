import logging
import time
from typing import Dict, Any, Optional, List
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Collect and expose system metrics for monitoring
    """
    
    def __init__(self, port: int = 9090):
        """
        Initialize the metrics collector
        
        Args:
            port: Port to expose Prometheus metrics on
        """
        self.port = port
        
        # Face detection metrics
        self.face_detection_counter = Counter(
            'face_detection_total',
            'Total number of face detections',
            ['camera_id', 'status']
        )
        
        self.face_detection_latency = Histogram(
            'face_detection_latency_seconds',
            'Face detection latency in seconds',
            ['camera_id'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Face recognition metrics
        self.face_recognition_counter = Counter(
            'face_recognition_total',
            'Total number of face recognition attempts',
            ['status']
        )
        
        self.face_recognition_latency = Histogram(
            'face_recognition_latency_seconds',
            'Face recognition latency in seconds',
            buckets=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.recognition_confidence = Histogram(
            'recognition_confidence',
            'Distribution of face recognition confidence scores',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        )
        
        # Database metrics
        self.database_operation_latency = Histogram(
            'database_operation_latency_seconds',
            'Database operation latency in seconds',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        self.database_size = Gauge(
            'database_size',
            'Number of faces in the database',
            ['shard_id']
        )
        
        # Camera metrics
        self.active_cameras = Gauge(
            'active_cameras',
            'Number of active cameras'
        )
        
        self.camera_fps = Gauge(
            'camera_fps',
            'Frames per second per camera',
            ['camera_id']
        )
        
        self.camera_errors = Counter(
            'camera_errors_total',
            'Total number of camera errors',
            ['camera_id', 'error_type']
        )
        
        # API metrics
        self.api_requests = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['endpoint', 'method', 'status']
        )
        
        self.api_latency = Histogram(
            'api_latency_seconds',
            'API request latency in seconds',
            ['endpoint', 'method'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # System metrics
        self.system_memory = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes'
        )
        
        self.system_cpu = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.gpu_memory = Gauge(
            'gpu_memory_usage_bytes',
            'GPU memory usage in bytes'
        )
        
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage'
        )
        
        # Start the metrics server
        try:
            start_http_server(port)
            logger.info(f"Started metrics server on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def record_face_detection(self, camera_id: str, success: bool, latency: float):
        """
        Record a face detection attempt
        
        Args:
            camera_id: Camera identifier
            success: Whether detection was successful
            latency: Detection latency in seconds
        """
        status = 'success' if success else 'failure'
        self.face_detection_counter.labels(camera_id=camera_id, status=status).inc()
        self.face_detection_latency.labels(camera_id=camera_id).observe(latency)
    
    def record_face_recognition(self, success: bool, latency: float, confidence: Optional[float] = None):
        """
        Record a face recognition attempt
        
        Args:
            success: Whether recognition was successful
            latency: Recognition latency in seconds
            confidence: Recognition confidence (if available)
        """
        status = 'success' if success else 'failure'
        self.face_recognition_counter.labels(status=status).inc()
        self.face_recognition_latency.observe(latency)
        
        if confidence is not None:
            self.recognition_confidence.observe(confidence)
    
    def record_database_operation(self, operation: str, latency: float):
        """
        Record a database operation
        
        Args:
            operation: Operation type (e.g., 'add', 'query', 'update', 'delete')
            latency: Operation latency in seconds
        """
        self.database_operation_latency.labels(operation=operation).observe(latency)
    
    def update_database_size(self, shard_id: str, size: int):
        """
        Update the database size metric
        
        Args:
            shard_id: Shard identifier
            size: Number of faces in the shard
        """
        self.database_size.labels(shard_id=shard_id).set(size)
    
    def update_camera_status(self, camera_id: str, is_active: bool, fps: Optional[float] = None):
        """
        Update camera status metrics
        
        Args:
            camera_id: Camera identifier
            is_active: Whether the camera is active
            fps: Frames per second (if available)
        """
        # We'll use a class variable to track active cameras
        if not hasattr(self, '_active_cameras'):
            self._active_cameras = {}
            
        # Update status for this camera
        self._active_cameras[camera_id] = is_active
        
        # Update active cameras count
        active_count = sum(1 for status in self._active_cameras.values() if status)
        self.active_cameras.set(active_count)
        
        if fps is not None:
            self.camera_fps.labels(camera_id=camera_id).set(fps)
    
    def record_camera_error(self, camera_id: str, error_type: str):
        """
        Record a camera error
        
        Args:
            camera_id: Camera identifier
            error_type: Type of error (e.g., 'connection', 'decode', 'timeout')
        """
        self.camera_errors.labels(camera_id=camera_id, error_type=error_type).inc()
    
    def record_api_request(self, endpoint: str, method: str, status: int, latency: float):
        """
        Record an API request
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status: HTTP status code
            latency: Request latency in seconds
        """
        status_class = f"{status // 100}xx"
        self.api_requests.labels(endpoint=endpoint, method=method, status=status_class).inc()
        self.api_latency.labels(endpoint=endpoint, method=method).observe(latency)
    
    def update_system_metrics(self, memory_usage: int, cpu_usage: float,
                             gpu_memory: Optional[int] = None, gpu_utilization: Optional[float] = None):
        """
        Update system resource metrics
        
        Args:
            memory_usage: Memory usage in bytes
            cpu_usage: CPU usage percentage
            gpu_memory: GPU memory usage in bytes (if available)
            gpu_utilization: GPU utilization percentage (if available)
        """
        self.system_memory.set(memory_usage)
        self.system_cpu.set(cpu_usage)
        
        if gpu_memory is not None:
            self.gpu_memory.set(gpu_memory)
            
        if gpu_utilization is not None:
            self.gpu_utilization.set(gpu_utilization)


class MetricsMiddleware:
    """
    HTTP middleware for collecting API metrics
    
    Compatible with Flask, FastAPI, and other WSGI/ASGI frameworks
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize the middleware
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics = metrics_collector
    
    def flask_before_request(self):
        """Flask before_request handler"""
        # Store start time in g
        from flask import g
        g.start_time = time.time()
    
    def flask_after_request(self, response):
        """Flask after_request handler"""
        from flask import request, g
        
        latency = time.time() - g.start_time
        endpoint = request.endpoint or request.path
        method = request.method
        status = response.status_code
        
        self.metrics.record_api_request(endpoint, method, status, latency)
        
        return response
    
    def starlette_middleware(self):
        """Return a Starlette/FastAPI middleware"""
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import Response
        
        metrics = self.metrics
        
        class PrometheusMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                start_time = time.time()
                
                try:
                    response = await call_next(request)
                except Exception as exc:
                    # Handle exception (log and reraise)
                    raise exc
                finally:
                    latency = time.time() - start_time
                    status_code = response.status_code if 'response' in locals() else 500
                    endpoint = request.url.path
                    method = request.method
                    
                    metrics.record_api_request(endpoint, method, status_code, latency)
                
                return response
        
        return PrometheusMiddleware