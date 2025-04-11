import os
import numpy as np
import faiss
import uuid
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from models import Base, FaceRecord, RecognitionLog, CameraConfig

class Database:
    def __init__(self, db_url=None):
        # Get database URL from environment or use provided URL
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        if not self.db_url:
            raise ValueError("Database URL not provided and DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.db_url)
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        # Initialize FAISS index
        self.index = None
        self.face_ids = []
        self.rebuild_index()
        
        # Thread pool for background tasks
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logging.info("Database initialized")
    
    def rebuild_index(self):
        """Rebuild FAISS index from database"""
        session = self.Session()
        try:
            face_records = session.query(FaceRecord).all()
            embeddings = []
            face_ids = []
            
            for record in face_records:
                embeddings.append(record.get_embedding_array())
                face_ids.append(record.face_id)
            
            if embeddings:
                # Convert to numpy array
                embeddings_array = np.array(embeddings).astype('float32')
                dim = embeddings_array.shape[1]
                
                # Create new index
                self.index = faiss.IndexFlatL2(dim)
                self.index.add(embeddings_array)
                self.face_ids = face_ids
                logging.info(f"FAISS index rebuilt with {len(face_ids)} faces")
            else:
                self.index = None
                self.face_ids = []
                logging.info("No face embeddings found, FAISS index not created")
                
        except Exception as e:
            logging.error(f"Error rebuilding FAISS index: {e}")
        finally:
            session.close()
    
    def add_face(self, name, embedding, metadata=None, quality_score=None):
        """Add a new face to the database"""
        face_id = str(uuid.uuid4())
        session = self.Session()
        try:
            face_record = FaceRecord.from_embedding(
                face_id=face_id,
                name=name,
                embedding=embedding,
                metadata=metadata,
                quality_score=quality_score
            )
            session.add(face_record)
            session.commit()
            
            # Schedule index rebuild in background
            self.executor.submit(self.rebuild_index)
            
            return face_id
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error adding face: {e}")
            raise
        finally:
            session.close()
    
    def update_face(self, face_id, name=None, embedding=None, metadata=None, quality_score=None):
        """Update an existing face in the database"""
        session = self.Session()
        try:
            face_record = session.query(FaceRecord).filter_by(face_id=face_id).first()
            if not face_record:
                return False
            
            if name:
                face_record.name = name
            
            if embedding is not None:
                if isinstance(embedding, np.ndarray):
                    face_record.embedding = embedding.astype(np.float32).tobytes()
                else:
                    face_record.embedding = np.array(embedding, dtype=np.float32).tobytes()
            
            if metadata:
                if face_record.metadata:
                    face_record.metadata.update(metadata)
                else:
                    face_record.metadata = metadata
            
            if quality_score is not None:
                face_record.quality_score = quality_score
            
            face_record.updated_at = datetime.utcnow()
            session.commit()
            
            # Schedule index rebuild in background if embedding changed
            if embedding is not None:
                self.executor.submit(self.rebuild_index)
            
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error updating face: {e}")
            raise
        finally:
            session.close()
    
    def delete_face(self, face_id):
        """Delete a face from the database"""
        session = self.Session()
        try:
            face_record = session.query(FaceRecord).filter_by(face_id=face_id).first()
            if not face_record:
                return False
            
            session.delete(face_record)
            session.commit()
            
            # Schedule index rebuild in background
            self.executor.submit(self.rebuild_index)
            
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error deleting face: {e}")
            raise
        finally:
            session.close()
    
    def get_face(self, face_id):
        """Get a face by ID"""
        session = self.Session()
        try:
            face_record = session.query(FaceRecord).filter_by(face_id=face_id).first()
            return face_record
        finally:
            session.close()
    
    def get_all_faces(self, offset=0, limit=100):
        """Get all faces with pagination"""
        session = self.Session()
        try:
            total = session.query(FaceRecord).count()
            faces = session.query(FaceRecord).order_by(FaceRecord.created_at.desc()).offset(offset).limit(limit).all()
            return faces, total
        finally:
            session.close()
    
    def find_similar_faces(self, embedding, threshold=0.6, max_results=5):
        """Find similar faces using FAISS vector search"""
        if self.index is None or not self.face_ids:
            return []
        
        # Convert embedding to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Reshape to expected format
        embedding = embedding.reshape(1, -1).astype('float32')
        
        # Search using FAISS
        distances, indices = self.index.search(embedding, min(max_results, len(self.face_ids)))
        
        results = []
        session = self.Session()
        try:
            for i, idx in enumerate(indices[0]):
                if idx == -1 or distances[0][i] > threshold:
                    continue
                
                face_id = self.face_ids[idx]
                face = session.query(FaceRecord).filter_by(face_id=face_id).first()
                if face:
                    confidence = 1.0 - (distances[0][i] / threshold)
                    results.append({
                        'face': face,
                        'confidence': float(confidence),
                        'distance': float(distances[0][i])
                    })
            
            return results
        finally:
            session.close()
    
    def log_recognition(self, face_id, camera_id, confidence, details=None):
        """Log a recognition event"""
        session = self.Session()
        try:
            log = RecognitionLog(
                face_id=face_id,
                camera_id=camera_id,
                confidence=confidence,
                details=details or {}
            )
            session.add(log)
            session.commit()
            return log.id
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error logging recognition: {e}")
            raise
        finally:
            session.close()
    
    def get_recognition_logs(self, offset=0, limit=100, face_id=None, camera_id=None):
        """Get recognition logs with optional filtering"""
        session = self.Session()
        try:
            query = session.query(RecognitionLog)
            
            if face_id:
                query = query.filter_by(face_id=face_id)
            if camera_id:
                query = query.filter_by(camera_id=camera_id)
            
            total = query.count()
            logs = query.order_by(RecognitionLog.timestamp.desc()).offset(offset).limit(limit).all()
            
            return logs, total
        finally:
            session.close()
    
    def add_or_update_camera(self, camera_id, name, url=None, device_id=None, enabled=True, config=None):
        """Add or update camera configuration"""
        session = self.Session()
        try:
            camera = session.query(CameraConfig).filter_by(camera_id=camera_id).first()
            
            if camera:
                camera.name = name
                if url is not None:
                    camera.url = url
                if device_id is not None:
                    camera.device_id = device_id
                camera.enabled = 1 if enabled else 0
                if config is not None:
                    camera.config = config
            else:
                camera = CameraConfig(
                    camera_id=camera_id,
                    name=name,
                    url=url,
                    device_id=device_id,
                    enabled=1 if enabled else 0,
                    config=config or {}
                )
                session.add(camera)
            
            session.commit()
            return camera.id
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error adding/updating camera: {e}")
            raise
        finally:
            session.close()
    
    def get_cameras(self, only_enabled=False):
        """Get all camera configurations"""
        session = self.Session()
        try:
            query = session.query(CameraConfig)
            
            if only_enabled:
                query = query.filter_by(enabled=1)
            
            cameras = query.all()
            return cameras
        finally:
            session.close()
    
    def get_camera(self, camera_id):
        """Get a camera configuration by ID"""
        session = self.Session()
        try:
            camera = session.query(CameraConfig).filter_by(camera_id=camera_id).first()
            return camera
        finally:
            session.close()
    
    def delete_camera(self, camera_id):
        """Delete a camera configuration"""
        session = self.Session()
        try:
            camera = session.query(CameraConfig).filter_by(camera_id=camera_id).first()
            if not camera:
                return False
            
            session.delete(camera)
            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error deleting camera: {e}")
            raise
        finally:
            session.close()
