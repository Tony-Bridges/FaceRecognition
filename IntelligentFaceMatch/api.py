import os
import grpc
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Import proto-generated code
from proto import face_recognition_pb2
from proto import face_recognition_pb2_grpc

class ApiClient:
    """
    gRPC client for the face recognition service
    Provides async methods for all API operations
    """
    
    def __init__(self, server_address: str):
        """
        Initialize the API client
        
        Args:
            server_address: gRPC server address (host:port)
        """
        self.server_address = server_address
        self.channel = None
        self.stub = None
        
        # Connect to gRPC server
        self._connect()
        
        logging.info(f"API client initialized with server: {server_address}")
    
    def _connect(self):
        """Connect to gRPC server"""
        # Options for larger messages
        options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]
        
        # Create channel and stub
        self.channel = grpc.aio.insecure_channel(self.server_address, options=options)
        self.stub = face_recognition_pb2_grpc.FaceRecognitionServiceStub(self.channel)
    
    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
    
    def _create_metadata(self, metadata_dict: Dict) -> face_recognition_pb2.FaceMetadata:
        """Create FaceMetadata proto from dictionary"""
        metadata = face_recognition_pb2.FaceMetadata()
        
        # Set basic fields
        if 'registered_by' in metadata_dict:
            metadata.registered_by = metadata_dict['registered_by']
        
        if 'device_id' in metadata_dict:
            metadata.device_id = metadata_dict['device_id']
        
        if 'registration_date' in metadata_dict:
            metadata.registration_date = metadata_dict['registration_date']
        
        if 'last_accessed' in metadata_dict:
            metadata.last_accessed = metadata_dict['last_accessed']
        
        # Add additional info
        for key, value in metadata_dict.items():
            if key not in ['registered_by', 'device_id', 'registration_date', 'last_accessed']:
                if isinstance(value, str):
                    metadata.additional_info[key] = value
                else:
                    # Convert non-string values to string
                    metadata.additional_info[key] = str(value)
        
        return metadata
    
    async def register_face(self, image_bytes: bytes, name: str, verify_liveness: bool = False, metadata: Dict = None) -> face_recognition_pb2.RegisterFaceResponse:
        """
        Register a face in the database
        
        Args:
            image_bytes: Image containing a face
            name: Name for the face
            verify_liveness: Whether to verify liveness
            metadata: Additional metadata
            
        Returns:
            RegisterFaceResponse
        """
        # Create request
        request = face_recognition_pb2.RegisterFaceRequest(
            image=image_bytes,
            name=name,
            verify_liveness=verify_liveness
        )
        
        # Add metadata if provided
        if metadata:
            request.metadata.CopyFrom(self._create_metadata(metadata))
        
        # Send request
        return await self.stub.RegisterFace(request)
    
    async def recognize_faces(self, image_bytes: bytes, threshold: float = 0.6, max_results: int = 5) -> face_recognition_pb2.RecognizeResponse:
        """
        Recognize faces in an image
        
        Args:
            image_bytes: Image to analyze
            threshold: Recognition confidence threshold
            max_results: Maximum number of results
            
        Returns:
            RecognizeResponse
        """
        # Create request
        request = face_recognition_pb2.RecognizeRequest(
            image=image_bytes,
            threshold=threshold,
            max_results=max_results
        )
        
        # Send request
        return await self.stub.RecognizeFaces(request)
    
    async def list_faces(self, offset: int = 0, limit: int = 100) -> face_recognition_pb2.ListFacesResponse:
        """
        Get all registered faces
        
        Args:
            offset: Pagination offset
            limit: Maximum number of faces to return
            
        Returns:
            ListFacesResponse
        """
        # Create request
        request = face_recognition_pb2.ListFacesRequest(
            offset=offset,
            limit=limit
        )
        
        # Send request
        return await self.stub.ListFaces(request)
    
    async def delete_face(self, face_id: str) -> face_recognition_pb2.DeleteFaceResponse:
        """
        Delete a face from the database
        
        Args:
            face_id: ID of the face to delete
            
        Returns:
            DeleteFaceResponse
        """
        # Create request
        request = face_recognition_pb2.DeleteFaceRequest(
            face_id=face_id
        )
        
        # Send request
        return await self.stub.DeleteFace(request)
    
    async def update_face_metadata(self, face_id: str, metadata: Dict) -> face_recognition_pb2.UpdateMetadataResponse:
        """
        Update face metadata
        
        Args:
            face_id: ID of the face to update
            metadata: New metadata
            
        Returns:
            UpdateMetadataResponse
        """
        # Create request
        request = face_recognition_pb2.UpdateMetadataRequest(
            face_id=face_id,
            metadata=self._create_metadata(metadata)
        )
        
        # Send request
        return await self.stub.UpdateFaceMetadata(request)
    
    async def get_face_quality(self, image_bytes: bytes) -> face_recognition_pb2.FaceQualityResponse:
        """
        Get face quality score
        
        Args:
            image_bytes: Image containing a face
            
        Returns:
            FaceQualityResponse
        """
        # Create request
        request = face_recognition_pb2.FaceQualityRequest(
            image=image_bytes
        )
        
        # Send request
        return await self.stub.GetFaceQuality(request)
    
    async def verify_liveness(self, image_bytes: bytes, session_id: str) -> face_recognition_pb2.LivenessResponse:
        """
        Verify liveness of face
        
        Args:
            image_bytes: Image containing a face
            session_id: Session ID for liveness check
            
        Returns:
            LivenessResponse
        """
        # Create request
        request = face_recognition_pb2.LivenessRequest(
            image=image_bytes,
            session_id=session_id
        )
        
        # Send request
        return await self.stub.VerifyLiveness(request)
