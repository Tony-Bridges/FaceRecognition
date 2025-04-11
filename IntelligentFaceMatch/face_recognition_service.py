import io
import cv2
import numpy as np
import uuid
import grpc
import logging
import time
import os
from concurrent import futures
from datetime import datetime

# Import proto-generated code
# In a real implementation, these would be generated from the .proto file
# Here we'll define stubs for demonstration
from database import Database
from face_quality import FaceQualityAnalyzer
from recognition.detector import FaceDetector
from recognition.face_encoder import FaceEncoder

class FaceRecognitionServicer:
    """
    Implementation of the gRPC FaceRecognitionService
    """
    
    def __init__(self, db_url=None):
        # Initialize components
        self.database = Database(db_url)
        self.face_detector = FaceDetector(backend="opencv")
        self.face_encoder = FaceEncoder(backend="insightface" if self._check_insightface() else "opencv")
        self.quality_analyzer = FaceQualityAnalyzer()
        
        # Session tracking for liveness detection
        self.liveness_sessions = {}
        
        logging.info("Face Recognition Service initialized")
    
    def _check_insightface(self):
        """Check if InsightFace is available"""
        try:
            import insightface
            return True
        except ImportError:
            return False
    
    def RegisterFace(self, request, context):
        """Register a face in the database"""
        try:
            # Convert image bytes to OpenCV format
            image_bytes = request.image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid image data")
                return self._create_register_response(False, "", "Invalid image data")
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if not faces:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("No face detected in the image")
                return self._create_register_response(False, "", "No face detected")
            
            if len(faces) > 1:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Multiple faces detected, please provide an image with a single face")
                return self._create_register_response(False, "", "Multiple faces detected")
            
            face = faces[0]
            x, y, w, h = face['rect']
            
            # Extract face region
            face_img = image[y:y+h, x:x+w]
            
            # Analyze face quality
            quality_metrics = self.quality_analyzer.analyze_face(image, face['rect'], face['landmarks'])
            
            # Check if quality is acceptable
            if quality_metrics.overall_score < 0.5:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(f"Face quality too low ({quality_metrics.overall_score:.2f})")
                return self._create_register_response(False, "", f"Face quality too low: {quality_metrics.overall_score:.2f}")
            
            # Encode face
            face_encoding = self.face_encoder.encode_face(face_img)
            
            if face_encoding is None:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Failed to encode face")
                return self._create_register_response(False, "", "Failed to encode face")
            
            # Convert metadata from request
            metadata = self._parse_metadata(request.metadata) if hasattr(request, 'metadata') else None
            
            # Add to database
            face_id = self.database.add_face(
                name=request.name,
                embedding=face_encoding,
                metadata=metadata,
                quality_score=quality_metrics.overall_score
            )
            
            logging.info(f"Face registered with ID: {face_id}")
            
            return self._create_register_response(
                True, 
                face_id, 
                "Face registered successfully", 
                quality_metrics.overall_score
            )
        
        except Exception as e:
            logging.error(f"Error registering face: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return self._create_register_response(False, "", f"Error: {str(e)}")
    
    def RecognizeFaces(self, request, context):
        """Recognize faces in an image"""
        try:
            # Convert image bytes to OpenCV format
            image_bytes = request.image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid image data")
                return self._create_recognize_response([])
            
            # Get parameters
            threshold = request.threshold if hasattr(request, 'threshold') and request.threshold > 0 else 0.6
            max_results = request.max_results if hasattr(request, 'max_results') and request.max_results > 0 else 5
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if not faces:
                return self._create_recognize_response([])
            
            # Process each face
            matches = []
            for face in faces:
                x, y, w, h = face['rect']
                
                # Extract face region
                face_img = image[y:y+h, x:x+w]
                
                # Skip if invalid face image
                if face_img.size == 0:
                    continue
                
                # Encode face
                face_encoding = self.face_encoder.encode_face(face_img)
                
                if face_encoding is None:
                    continue
                
                # Find similar faces
                similar_faces = self.database.find_similar_faces(
                    embedding=face_encoding,
                    threshold=threshold,
                    max_results=max_results
                )
                
                # Add matches
                for result in similar_faces:
                    db_face = result['face']
                    confidence = result['confidence']
                    
                    # Create match object
                    face_object = self._create_face_object(
                        db_face.face_id,
                        db_face.name,
                        db_face.get_embedding_array().tolist(),
                        (x, y, w, h),
                        db_face.metadata,
                        db_face.quality_score
                    )
                    
                    matches.append(self._create_face_match(face_object, confidence))
                    
                    # Log recognition event
                    self.database.log_recognition(
                        face_id=db_face.face_id,
                        camera_id="api_request",
                        confidence=confidence,
                        details={
                            "timestamp": datetime.now().isoformat(),
                            "source": "api",
                            "rect": [x, y, w, h]
                        }
                    )
            
            return self._create_recognize_response(matches)
        
        except Exception as e:
            logging.error(f"Error recognizing faces: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return self._create_recognize_response([])
    
    def ListFaces(self, request, context):
        """Get all registered faces"""
        try:
            offset = request.offset if hasattr(request, 'offset') else 0
            limit = request.limit if hasattr(request, 'limit') else 100
            
            faces, total = self.database.get_all_faces(offset, limit)
            
            face_objects = []
            for face in faces:
                face_obj = self._create_face_object(
                    face.face_id,
                    face.name,
                    face.get_embedding_array().tolist(),
                    None,  # No rect for stored faces
                    face.metadata,
                    face.quality_score
                )
                face_objects.append(face_obj)
            
            return self._create_list_faces_response(face_objects, total)
        
        except Exception as e:
            logging.error(f"Error listing faces: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return self._create_list_faces_response([], 0)
    
    def DeleteFace(self, request, context):
        """Delete a face from the database"""
        try:
            face_id = request.face_id
            
            success = self.database.delete_face(face_id)
            
            if success:
                return self._create_delete_response(True, "Face deleted successfully")
            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Face with ID {face_id} not found")
                return self._create_delete_response(False, f"Face with ID {face_id} not found")
        
        except Exception as e:
            logging.error(f"Error deleting face: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return self._create_delete_response(False, f"Error: {str(e)}")
    
    def UpdateFaceMetadata(self, request, context):
        """Update face metadata"""
        try:
            face_id = request.face_id
            metadata = self._parse_metadata(request.metadata)
            
            success = self.database.update_face(face_id, metadata=metadata)
            
            if success:
                return self._create_update_metadata_response(True, "Metadata updated successfully")
            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Face with ID {face_id} not found")
                return self._create_update_metadata_response(False, f"Face with ID {face_id} not found")
        
        except Exception as e:
            logging.error(f"Error updating metadata: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return self._create_update_metadata_response(False, f"Error: {str(e)}")
    
    def GetFaceQuality(self, request, context):
        """Get face quality score"""
        try:
            # Convert image bytes to OpenCV format
            image_bytes = request.image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid image data")
                return self._create_quality_response(0.0, {})
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if not faces:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("No face detected in the image")
                return self._create_quality_response(0.0, {})
            
            if len(faces) > 1:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Multiple faces detected, please provide an image with a single face")
                return self._create_quality_response(0.0, {})
            
            face = faces[0]
            
            # Analyze face quality
            quality_metrics = self.quality_analyzer.analyze_face(image, face['rect'], face['landmarks'])
            
            # Convert quality factors to dict
            quality_factors = quality_metrics.to_dict()
            
            return self._create_quality_response(quality_metrics.overall_score, quality_factors)
        
        except Exception as e:
            logging.error(f"Error analyzing face quality: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return self._create_quality_response(0.0, {})
    
    def VerifyLiveness(self, request, context):
        """Verify liveness of face"""
        try:
            session_id = request.session_id
            
            # Convert image bytes to OpenCV format
            image_bytes = request.image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid image data")
                return self._create_liveness_response(False, 0.0, "Invalid image data")
            
            # Initialize session if new
            if session_id not in self.liveness_sessions:
                self.liveness_sessions[session_id] = {
                    'frames': [],
                    'last_updated': time.time(),
                    'blink_detected': False,
                    'head_moved': False,
                    'status': 'initial'
                }
            
            session = self.liveness_sessions[session_id]
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if not faces:
                return self._create_liveness_response(False, 0.0, "No face detected")
            
            if len(faces) > 1:
                return self._create_liveness_response(False, 0.0, "Multiple faces detected")
            
            face = faces[0]
            
            # Add frame to session
            session['frames'].append({
                'timestamp': time.time(),
                'face_rect': face['rect'],
                'landmarks': face['landmarks']
            })
            session['last_updated'] = time.time()
            
            # Simple liveness check (example)
            # In a real system, you'd have more sophisticated checks
            is_live = False
            confidence = 0.0
            message = "Liveness check in progress"
            
            # Simple implementation - just assume live after a few frames
            # In reality, you'd check for blink detection, head movement, etc.
            if len(session['frames']) >= 3:
                is_live = True
                confidence = 0.8
                message = "Liveness confirmed"
            
            # Clean up old sessions
            self._cleanup_sessions()
            
            return self._create_liveness_response(is_live, confidence, message)
        
        except Exception as e:
            logging.error(f"Error verifying liveness: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return self._create_liveness_response(False, 0.0, f"Error: {str(e)}")
    
    def _cleanup_sessions(self):
        """Clean up old liveness sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.liveness_sessions.items():
            if current_time - session['last_updated'] > 300:  # 5 minutes timeout
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.liveness_sessions[session_id]
    
    def _parse_metadata(self, proto_metadata):
        """Parse metadata from proto to dict"""
        metadata = {}
        
        if hasattr(proto_metadata, 'registered_by'):
            metadata['registered_by'] = proto_metadata.registered_by
        
        if hasattr(proto_metadata, 'device_id'):
            metadata['device_id'] = proto_metadata.device_id
        
        if hasattr(proto_metadata, 'registration_date'):
            metadata['registration_date'] = proto_metadata.registration_date
        else:
            metadata['registration_date'] = datetime.now().isoformat()
        
        if hasattr(proto_metadata, 'last_accessed'):
            metadata['last_accessed'] = proto_metadata.last_accessed
        
        if hasattr(proto_metadata, 'additional_info'):
            for key, value in proto_metadata.additional_info.items():
                metadata[key] = value
        
        return metadata
    
    # The following methods would normally use the generated proto classes
    # Here we're creating mock response objects for demonstration
    
    def _create_register_response(self, success, face_id, message, quality_score=0.0):
        """Create a RegisterFaceResponse (mock for demonstration)"""
        # In a real implementation, this would use the generated proto class
        class RegisterFaceResponse:
            def __init__(self, success, face_id, message, quality_score):
                self.success = success
                self.face_id = face_id
                self.message = message
                self.quality_score = quality_score
        
        return RegisterFaceResponse(success, face_id, message, quality_score)
    
    def _create_face_object(self, face_id, name, embedding, rect, metadata, quality_score):
        """Create a Face object (mock for demonstration)"""
        # In a real implementation, this would use the generated proto class
        class Face:
            def __init__(self, face_id, name, embedding, rect, metadata, quality_score):
                self.face_id = face_id
                self.name = name
                self.embedding = embedding
                self.rect = self._create_rect(rect)
                self.metadata = self._create_metadata(metadata)
                self.quality_score = quality_score
            
            def _create_rect(self, rect):
                if rect is None:
                    return None
                
                class FaceRect:
                    def __init__(self, x, y, width, height):
                        self.x = x
                        self.y = y
                        self.width = width
                        self.height = height
                
                return FaceRect(rect[0], rect[1], rect[2], rect[3])
            
            def _create_metadata(self, metadata):
                class FaceMetadata:
                    def __init__(self, data):
                        self.registered_by = data.get('registered_by', '')
                        self.device_id = data.get('device_id', '')
                        self.registration_date = data.get('registration_date', '')
                        self.last_accessed = data.get('last_accessed', '')
                        self.additional_info = data.copy()
                
                return FaceMetadata(metadata or {})
        
        return Face(face_id, name, embedding, rect, metadata, quality_score)
    
    def _create_face_match(self, face, confidence):
        """Create a FaceMatch object (mock for demonstration)"""
        class FaceMatch:
            def __init__(self, face, confidence):
                self.face = face
                self.confidence = confidence
        
        return FaceMatch(face, confidence)
    
    def _create_recognize_response(self, matches):
        """Create a RecognizeResponse (mock for demonstration)"""
        class RecognizeResponse:
            def __init__(self, matches):
                self.matches = matches
        
        return RecognizeResponse(matches)
    
    def _create_list_faces_response(self, faces, total):
        """Create a ListFacesResponse (mock for demonstration)"""
        class ListFacesResponse:
            def __init__(self, faces, total):
                self.faces = faces
                self.total = total
        
        return ListFacesResponse(faces, total)
    
    def _create_delete_response(self, success, message):
        """Create a DeleteFaceResponse (mock for demonstration)"""
        class DeleteFaceResponse:
            def __init__(self, success, message):
                self.success = success
                self.message = message
        
        return DeleteFaceResponse(success, message)
    
    def _create_update_metadata_response(self, success, message):
        """Create an UpdateMetadataResponse (mock for demonstration)"""
        class UpdateMetadataResponse:
            def __init__(self, success, message):
                self.success = success
                self.message = message
        
        return UpdateMetadataResponse(success, message)
    
    def _create_quality_response(self, quality_score, quality_factors):
        """Create a FaceQualityResponse (mock for demonstration)"""
        class FaceQualityResponse:
            def __init__(self, quality_score, quality_factors):
                self.quality_score = quality_score
                self.quality_factors = quality_factors
        
        return FaceQualityResponse(quality_score, quality_factors)
    
    def _create_liveness_response(self, is_live, confidence, message):
        """Create a LivenessResponse (mock for demonstration)"""
        class LivenessResponse:
            def __init__(self, is_live, confidence, message):
                self.is_live = is_live
                self.confidence = confidence
                self.message = message
        
        return LivenessResponse(is_live, confidence, message)
