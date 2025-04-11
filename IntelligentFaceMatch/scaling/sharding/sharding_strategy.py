import hashlib
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

class ShardingStrategy(ABC):
    """
    Abstract base class for sharding strategies
    """
    
    @abstractmethod
    def get_shard_for_face(self, name: str, embedding: np.ndarray, 
                          metadata: Optional[Dict] = None, num_shards: int = 1) -> int:
        """
        Determine which shard a face should be stored in
        
        Args:
            name: Person's name
            embedding: Face embedding vector
            metadata: Additional metadata about the face
            num_shards: Total number of available shards
            
        Returns:
            Shard ID (0-based index)
        """
        pass


class UniformShardingStrategy(ShardingStrategy):
    """
    Simple uniform distribution strategy based on hash of face ID or name
    """
    
    def get_shard_for_face(self, name: str, embedding: np.ndarray, 
                          metadata: Optional[Dict] = None, num_shards: int = 1) -> int:
        # Use face_id if available, otherwise use name
        face_id = metadata.get('face_id') if metadata else None
        key = face_id if face_id else name
        
        # Generate a hash of the key
        hash_value = int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
        
        # Distribute uniformly based on hash
        return hash_value % num_shards


class GeographicShardingStrategy(ShardingStrategy):
    """
    Geographic sharding based on location metadata
    """
    
    def __init__(self, region_mapping: Optional[Dict[str, int]] = None):
        """
        Initialize with region to shard mapping
        
        Args:
            region_mapping: Dictionary mapping region names to shard IDs
        """
        self.region_mapping = region_mapping or {}
    
    def get_shard_for_face(self, name: str, embedding: np.ndarray, 
                          metadata: Optional[Dict] = None, num_shards: int = 1) -> int:
        if not metadata:
            return 0
        
        # Check for location information in metadata
        location = metadata.get('location') or metadata.get('region')
        if not location:
            # Fall back to uniform strategy
            return UniformShardingStrategy().get_shard_for_face(name, embedding, metadata, num_shards)
        
        # Use the region mapping if available
        if location in self.region_mapping:
            shard_id = self.region_mapping[location]
            if 0 <= shard_id < num_shards:
                return shard_id
        
        # Hash-based mapping for regions not explicitly configured
        hash_value = int(hashlib.md5(location.encode('utf-8')).hexdigest(), 16)
        return hash_value % num_shards


class CameraGroupShardingStrategy(ShardingStrategy):
    """
    Shard faces based on the camera group they were captured with
    """
    
    def __init__(self, camera_group_mapping: Optional[Dict[str, int]] = None):
        """
        Initialize with camera group to shard mapping
        
        Args:
            camera_group_mapping: Dictionary mapping camera group IDs to shard IDs
        """
        self.camera_group_mapping = camera_group_mapping or {}
    
    def get_shard_for_face(self, name: str, embedding: np.ndarray, 
                          metadata: Optional[Dict] = None, num_shards: int = 1) -> int:
        if not metadata:
            return 0
        
        # Check for camera information in metadata
        camera_id = metadata.get('camera_id')
        camera_group = metadata.get('camera_group')
        
        # Priority: try camera_group, then camera_id
        group_key = camera_group if camera_group else camera_id
        
        if group_key and group_key in self.camera_group_mapping:
            shard_id = self.camera_group_mapping[group_key]
            if 0 <= shard_id < num_shards:
                return shard_id
        
        # If no mapping or no camera info, use uniform distribution
        if not group_key:
            return UniformShardingStrategy().get_shard_for_face(name, embedding, metadata, num_shards)
        
        # Hash-based mapping for groups not explicitly configured
        hash_value = int(hashlib.md5(str(group_key).encode('utf-8')).hexdigest(), 16)
        return hash_value % num_shards