import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database settings
DATABASE_URL = os.environ.get("DATABASE_URL")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Face recognition settings
FACE_RECOGNITION_MODEL = os.environ.get("FACE_RECOGNITION_MODEL", "insightface")
FACE_DETECTION_THRESHOLD = float(os.environ.get("FACE_DETECTION_THRESHOLD", "0.5"))
FACE_RECOGNITION_THRESHOLD = float(os.environ.get("FACE_RECOGNITION_THRESHOLD", "0.6"))
FACE_QUALITY_THRESHOLD = float(os.environ.get("FACE_QUALITY_THRESHOLD", "0.7"))
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"

# Camera settings
MAX_FRAME_RATE = int(os.environ.get("MAX_FRAME_RATE", "15"))
FRAME_SKIP_THRESHOLD = float(os.environ.get("FRAME_SKIP_THRESHOLD", "0.8"))  # CPU usage at which to start skipping frames

# gRPC server settings
GRPC_PORT = int(os.environ.get("GRPC_PORT", "8000"))
GRPC_HOST = os.environ.get("GRPC_HOST", "0.0.0.0")
GRPC_MAX_WORKERS = int(os.environ.get("GRPC_MAX_WORKERS", "10"))

# REST API settings
API_PORT = int(os.environ.get("API_PORT", "5000"))
API_HOST = os.environ.get("API_HOST", "0.0.0.0")

# MJPEG stream settings
MJPEG_PORT = int(os.environ.get("MJPEG_PORT", "8001"))
MJPEG_HOST = os.environ.get("MJPEG_HOST", "0.0.0.0")

# Advanced features
ENABLE_EMOTION_DETECTION = os.environ.get("ENABLE_EMOTION_DETECTION", "false").lower() == "true"
ENABLE_AGE_GENDER_ESTIMATION = os.environ.get("ENABLE_AGE_GENDER_ESTIMATION", "false").lower() == "true"
ENABLE_RE_IDENTIFICATION = os.environ.get("ENABLE_RE_IDENTIFICATION", "false").lower() == "true"
ENABLE_ACTIVE_LEARNING = os.environ.get("ENABLE_ACTIVE_LEARNING", "false").lower() == "true"

# Performance and scaling settings
ENABLE_WASM = os.environ.get("ENABLE_WASM", "false").lower() == "true"
SHARDING_ENABLED = os.environ.get("SHARDING_ENABLED", "false").lower() == "true"
SHARDING_STRATEGY = os.environ.get("SHARDING_STRATEGY", "geographic")  # geographic, camera_group, uniform
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
LOAD_BALANCING_STRATEGY = os.environ.get("LOAD_BALANCING_STRATEGY", "round_robin")  # round_robin, least_connections

# Monitoring settings
PROMETHEUS_ENABLED = os.environ.get("PROMETHEUS_ENABLED", "false").lower() == "true"
PROMETHEUS_PORT = int(os.environ.get("PROMETHEUS_PORT", "9090"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")