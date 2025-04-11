import os
from datetime import datetime
import json
import numpy as np

from database_setup import db

class FaceRecord(db.Model):
    """Database model for storing face recognition data"""
    __tablename__ = 'faces'
    
    id = db.Column(db.Integer, primary_key=True)
    face_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    name = db.Column(db.String(128), nullable=False, index=True)
    embedding = db.Column(db.LargeBinary, nullable=True)  # Stored as binary for efficiency
    quality_score = db.Column(db.Float, nullable=True)
    meta_data = db.Column(db.JSON, nullable=True)  # Changed from metadata to meta_data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'face_id': self.face_id,
            'name': self.name,
            'quality_score': self.quality_score,
            'metadata': self.meta_data,  # Keep the API consistent using 'metadata'
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def get_embedding_array(self):
        """Convert stored binary embedding to numpy array"""
        if self.embedding:
            return np.frombuffer(self.embedding, dtype=np.float32)
        return None
    
    @staticmethod
    def from_embedding(face_id, name, embedding=None, metadata=None, quality_score=None):
        """Create a FaceRecord from embedding array"""
        embedding_binary = None
        if embedding is not None:
            if isinstance(embedding, np.ndarray):
                embedding_binary = embedding.astype(np.float32).tobytes()
            else:
                embedding_binary = np.array(embedding, dtype=np.float32).tobytes()
            
        return FaceRecord(
            face_id=face_id,
            name=name,
            embedding=embedding_binary,
            meta_data=metadata or {},  # Changed from metadata to meta_data
            quality_score=quality_score
        )

class RecognitionLog(db.Model):
    """Database model for logging recognition events"""
    __tablename__ = 'recognition_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    face_id = db.Column(db.String(64), nullable=True)
    camera_id = db.Column(db.String(64), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    details = db.Column(db.JSON, nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'face_id': self.face_id,
            'camera_id': self.camera_id,
            'confidence': self.confidence,
            'details': self.details
        }

class CameraConfig(db.Model):
    """Database model for camera configuration"""
    __tablename__ = 'camera_configs'
    
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(128), nullable=False)
    url = db.Column(db.String(256), nullable=True)  # For IP cameras
    device_id = db.Column(db.Integer, nullable=True)  # For local webcams
    enabled = db.Column(db.Integer, default=1)
    config = db.Column(db.JSON, nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'name': self.name,
            'url': self.url,
            'device_id': self.device_id,
            'enabled': bool(self.enabled),
            'config': self.config
        }
