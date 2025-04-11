import logging
import threading
from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np

from .sharding_strategy import ShardingStrategy, UniformShardingStrategy

logger = logging.getLogger(__name__)

class ShardManager:
    """
    Manages distribution and retrieval of face embeddings across multiple database shards
    """
    
    def __init__(self, 
                 shard_urls: List[str], 
                 strategy: Optional[ShardingStrategy] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the shard manager
        
        Args:
            shard_urls: List of database URLs for each shard
            strategy: Sharding strategy (defaults to UniformShardingStrategy if None)
            config: Additional configuration options
        """
        self.shard_urls = shard_urls
        self.num_shards = len(shard_urls)
        self.strategy = strategy or UniformShardingStrategy()
        self.config = config or {}
        self.shard_connections = {}  # shard_id -> connection
        self.lock = threading.RLock()
        
        # Connect to all shards
        self._initialize_shards()
        
        logger.info(f"Initialized shard manager with {self.num_shards} shards using {self.strategy.__class__.__name__}")
    
    def _initialize_shards(self) -> None:
        """Initialize connections to all shards"""
        from database import Database  # Import here to avoid circular imports
        
        for i, url in enumerate(self.shard_urls):
            try:
                self.shard_connections[i] = Database(url)
                logger.info(f"Connected to shard {i} at {url}")
            except Exception as e:
                logger.error(f"Failed to connect to shard {i} at {url}: {e}")
    
    def add_face(self, name: str, embedding: np.ndarray, metadata: Dict = None, 
                quality_score: float = None) -> Tuple[bool, str, Optional[str]]:
        """
        Add a face to the appropriate shard based on the sharding strategy
        
        Args:
            name: Name of the person
            embedding: Face embedding vector
            metadata: Additional metadata
            quality_score: Face quality score
        
        Returns:
            Tuple of (success, face_id, error_message)
        """
        # Determine which shard to use
        shard_id = self.strategy.get_shard_for_face(name, embedding, metadata, self.num_shards)
        
        if shard_id not in self.shard_connections:
            error_msg = f"Shard {shard_id} not available"
            logger.error(error_msg)
            return False, None, error_msg
        
        # Add to the selected shard
        try:
            with self.lock:
                db = self.shard_connections[shard_id]
                success, face_id = db.add_face(name, embedding, metadata, quality_score)
                
                if success:
                    # Add shard info to metadata for retrieval
                    if metadata is None:
                        metadata = {}
                    metadata['_shard_id'] = shard_id
                    db.update_face(face_id, metadata=metadata)
                    
                    logger.info(f"Added face {face_id} to shard {shard_id}")
                    return True, face_id, None
                else:
                    logger.error(f"Failed to add face to shard {shard_id}")
                    return False, None, "Failed to add face to database"
        except Exception as e:
            error_msg = f"Error adding face to shard {shard_id}: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def find_similar_faces(self, embedding: np.ndarray, threshold: float = 0.6, 
                          max_results: int = 5) -> List[Dict]:
        """
        Find similar faces across all shards
        
        Args:
            embedding: Query face embedding
            threshold: Similarity threshold
            max_results: Maximum number of results
        
        Returns:
            List of face matches sorted by similarity
        """
        all_results = []
        
        # Query all shards in parallel
        def query_shard(shard_id, db):
            try:
                results = db.find_similar_faces(embedding, threshold, max_results)
                # Add shard ID to each result
                for result in results:
                    result['_shard_id'] = shard_id
                return results
            except Exception as e:
                logger.error(f"Error querying shard {shard_id}: {str(e)}")
                return []
        
        threads = []
        for shard_id, db in self.shard_connections.items():
            thread = threading.Thread(target=lambda s_id=shard_id, d=db: 
                                     all_results.extend(query_shard(s_id, d)))
            threads.append(thread)
            thread.start()
        
        # Wait for all queries to complete
        for thread in threads:
            thread.join()
        
        # Sort by similarity and truncate
        all_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return all_results[:max_results]
    
    def get_face(self, face_id: str) -> Optional[Dict]:
        """
        Get a face by ID from the appropriate shard
        
        Args:
            face_id: Face identifier
        
        Returns:
            Face information or None if not found
        """
        # Try to find metadata to determine shard
        for shard_id, db in self.shard_connections.items():
            face = db.get_face(face_id)
            if face:
                face['_shard_id'] = shard_id
                return face
        
        return None
    
    def delete_face(self, face_id: str) -> bool:
        """
        Delete a face from the appropriate shard
        
        Args:
            face_id: Face identifier
        
        Returns:
            True if successful
        """
        # First determine which shard contains the face
        for shard_id, db in self.shard_connections.items():
            face = db.get_face(face_id)
            if face:
                try:
                    success = db.delete_face(face_id)
                    if success:
                        logger.info(f"Deleted face {face_id} from shard {shard_id}")
                    else:
                        logger.error(f"Failed to delete face {face_id} from shard {shard_id}")
                    return success
                except Exception as e:
                    logger.error(f"Error deleting face {face_id} from shard {shard_id}: {str(e)}")
                    return False
        
        logger.warning(f"Face {face_id} not found in any shard")
        return False
    
    def get_all_faces(self, offset: int = 0, limit: int = 100) -> Tuple[List[Dict], int]:
        """
        Get all faces across all shards
        
        Args:
            offset: Pagination offset
            limit: Maximum number of faces to return
        
        Returns:
            Tuple of (faces list, total count)
        """
        all_faces = []
        total_count = 0
        
        # Gather faces from all shards
        for shard_id, db in self.shard_connections.items():
            try:
                faces, count = db.get_all_faces(0, 10000)  # Get all to count
                total_count += count
                
                # Add shard ID to each face
                for face in faces:
                    face['_shard_id'] = shard_id
                
                all_faces.extend(faces)
            except Exception as e:
                logger.error(f"Error getting faces from shard {shard_id}: {str(e)}")
        
        # Sort and paginate
        all_faces.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Apply pagination
        paginated_faces = all_faces[offset:offset+limit] if all_faces else []
        
        return paginated_faces, total_count
    
    def rebuild_indexes(self) -> bool:
        """
        Rebuild search indexes on all shards
        
        Returns:
            True if all successful
        """
        success = True
        
        for shard_id, db in self.shard_connections.items():
            try:
                if not db.rebuild_index():
                    success = False
                    logger.error(f"Failed to rebuild index for shard {shard_id}")
            except Exception as e:
                success = False
                logger.error(f"Error rebuilding index for shard {shard_id}: {str(e)}")
        
        return success
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the shards
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "num_shards": self.num_shards,
            "active_shards": len(self.shard_connections),
            "sharding_strategy": self.strategy.__class__.__name__,
            "per_shard_stats": {}
        }
        
        total_faces = 0
        for shard_id, db in self.shard_connections.items():
            try:
                _, count = db.get_all_faces(0, 1)
                stats["per_shard_stats"][f"shard_{shard_id}"] = {
                    "face_count": count,
                    "url": self.shard_urls[shard_id]
                }
                total_faces += count
            except Exception as e:
                logger.error(f"Error getting statistics from shard {shard_id}: {str(e)}")
                stats["per_shard_stats"][f"shard_{shard_id}"] = {
                    "error": str(e),
                    "url": self.shard_urls[shard_id]
                }
        
        stats["total_faces"] = total_faces
        
        return stats