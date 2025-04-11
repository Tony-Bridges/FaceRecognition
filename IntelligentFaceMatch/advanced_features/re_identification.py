import logging
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

class PersonReIdentification:
    """
    Re-identify people across different camera views and track their appearances over time
    """
    
    def __init__(self, 
                similarity_threshold: float = 0.7,
                tracking_timeout: int = 300,  # seconds
                max_history: int = 1000):
        """
        Initialize the re-identification module
        
        Args:
            similarity_threshold: Threshold for matching identity across cameras
            tracking_timeout: How long to keep track of a person without new sightings (seconds)
            max_history: Maximum number of sightings to store per person
        """
        self.similarity_threshold = similarity_threshold
        self.tracking_timeout = tracking_timeout
        self.max_history = max_history
        
        # Track people across cameras
        # person_id -> {
        #   'last_seen': timestamp,
        #   'cameras': set of camera_ids,
        #   'sightings': list of {camera_id, timestamp, face_id, confidence, location}
        # }
        self.people_tracking = {}
        
        # Track embeddings for quick matching
        # person_id -> list of embeddings
        self.person_embeddings = {}
        
        logger.info(f"Initialized person re-identification module with threshold {similarity_threshold}")
    
    def update(self, face_id: str, person_id: Optional[str], embedding: np.ndarray,
              camera_id: str, confidence: float, location: Optional[Dict] = None) -> str:
        """
        Update tracking with a new face detection
        
        Args:
            face_id: Unique ID of the detected face
            person_id: Known person ID (if already registered) or None
            embedding: Face embedding vector
            camera_id: Camera ID where the face was detected
            confidence: Detection confidence
            location: Optional location metadata
            
        Returns:
            Person ID (existing or new one)
        """
        # Clean up old tracking entries first
        self._cleanup_expired()
        
        # If person_id is provided, update tracking for that person
        if person_id and person_id in self.people_tracking:
            self._update_existing_person(person_id, face_id, embedding, camera_id, confidence, location)
            return person_id
        
        # Try to match with existing people
        matched_person_id = self._find_matching_person(embedding)
        
        if matched_person_id:
            # Update existing person
            self._update_existing_person(matched_person_id, face_id, embedding, camera_id, confidence, location)
            logger.info(f"Re-identified person {matched_person_id} on camera {camera_id}")
            return matched_person_id
        
        # No match, create new person tracking entry
        if not person_id:
            # Generate a new unique ID if none provided
            person_id = f"person_{int(time.time())}_{hash(str(embedding.tobytes())[:8])}"
        
        self._create_new_person(person_id, face_id, embedding, camera_id, confidence, location)
        logger.info(f"Added new person {person_id} from camera {camera_id}")
        
        return person_id
    
    def _update_existing_person(self, person_id: str, face_id: str, embedding: np.ndarray,
                               camera_id: str, confidence: float, location: Optional[Dict]) -> None:
        """Update tracking for an existing person"""
        now = time.time()
        
        # Update last seen time
        self.people_tracking[person_id]['last_seen'] = now
        
        # Add camera to set of cameras where person was seen
        self.people_tracking[person_id]['cameras'].add(camera_id)
        
        # Add to sightings history
        sighting = {
            'face_id': face_id,
            'camera_id': camera_id,
            'timestamp': now,
            'confidence': confidence,
            'location': location
        }
        
        self.people_tracking[person_id]['sightings'].append(sighting)
        
        # Trim history if needed
        if len(self.people_tracking[person_id]['sightings']) > self.max_history:
            self.people_tracking[person_id]['sightings'] = \
                self.people_tracking[person_id]['sightings'][-self.max_history:]
        
        # Add embedding to person's embedding list for future matching
        if person_id in self.person_embeddings:
            self.person_embeddings[person_id].append(embedding)
            # Keep only the most recent embeddings
            if len(self.person_embeddings[person_id]) > 5:
                self.person_embeddings[person_id] = self.person_embeddings[person_id][-5:]
    
    def _create_new_person(self, person_id: str, face_id: str, embedding: np.ndarray,
                          camera_id: str, confidence: float, location: Optional[Dict]) -> None:
        """Create tracking entry for a new person"""
        now = time.time()
        
        self.people_tracking[person_id] = {
            'last_seen': now,
            'cameras': {camera_id},
            'sightings': [{
                'face_id': face_id,
                'camera_id': camera_id,
                'timestamp': now,
                'confidence': confidence,
                'location': location
            }]
        }
        
        self.person_embeddings[person_id] = [embedding]
    
    def _find_matching_person(self, embedding: np.ndarray) -> Optional[str]:
        """Find matching person for a given embedding"""
        best_match = None
        best_similarity = -1
        
        for person_id, embeddings in self.person_embeddings.items():
            # Calculate similarity with all stored embeddings for this person
            similarities = [self._calculate_similarity(embedding, stored_emb) 
                           for stored_emb in embeddings]
            
            # Use maximum similarity
            if similarities:
                max_similarity = max(similarities)
                if max_similarity > best_similarity and max_similarity >= self.similarity_threshold:
                    best_similarity = max_similarity
                    best_match = person_id
        
        return best_match
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate similarity between two embeddings (cosine similarity)"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def _cleanup_expired(self) -> None:
        """Remove tracking data for people not seen recently"""
        now = time.time()
        expired_ids = []
        
        for person_id, data in self.people_tracking.items():
            if now - data['last_seen'] > self.tracking_timeout:
                expired_ids.append(person_id)
        
        for person_id in expired_ids:
            if person_id in self.people_tracking:
                del self.people_tracking[person_id]
            if person_id in self.person_embeddings:
                del self.person_embeddings[person_id]
                
        if expired_ids:
            logger.info(f"Removed {len(expired_ids)} expired tracking entries")
    
    def get_person_history(self, person_id: str) -> Dict[str, Any]:
        """
        Get tracking history for a person
        
        Args:
            person_id: Person identifier
            
        Returns:
            Dictionary with tracking history or empty dict if not found
        """
        if person_id not in self.people_tracking:
            return {}
        
        person_data = self.people_tracking[person_id]
        
        return {
            'person_id': person_id,
            'cameras': list(person_data['cameras']),
            'sightings': sorted(person_data['sightings'], key=lambda x: x['timestamp'], reverse=True),
            'first_seen': person_data['sightings'][0]['timestamp'] if person_data['sightings'] else None,
            'last_seen': person_data['last_seen'],
            'total_sightings': len(person_data['sightings'])
        }
    
    def get_camera_history(self, camera_id: str) -> List[Dict[str, Any]]:
        """
        Get all people seen by a specific camera
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            List of people seen by the camera with timestamps
        """
        results = []
        
        for person_id, data in self.people_tracking.items():
            if camera_id in data['cameras']:
                # Filter sightings for this camera
                camera_sightings = [s for s in data['sightings'] if s['camera_id'] == camera_id]
                
                if camera_sightings:
                    results.append({
                        'person_id': person_id,
                        'sightings': sorted(camera_sightings, key=lambda x: x['timestamp'], reverse=True),
                        'first_seen': min(s['timestamp'] for s in camera_sightings),
                        'last_seen': max(s['timestamp'] for s in camera_sightings),
                        'total_sightings': len(camera_sightings)
                    })
        
        # Sort by most recently seen
        results.sort(key=lambda x: x['last_seen'], reverse=True)
        return results
    
    def get_cross_camera_movements(self, time_window: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Track people movements across cameras
        
        Args:
            time_window: Optional time window in seconds (default: no limit)
            
        Returns:
            List of cross-camera movements sorted by recency
        """
        now = time.time()
        movements = []
        
        for person_id, data in self.people_tracking.items():
            # Skip if only seen on one camera
            if len(data['cameras']) <= 1:
                continue
                
            # Get sightings sorted by time
            sightings = sorted(data['sightings'], key=lambda x: x['timestamp'])
            
            # Filter by time window if specified
            if time_window:
                sightings = [s for s in sightings if now - s['timestamp'] <= time_window]
                
            # Find camera transitions
            for i in range(1, len(sightings)):
                if sightings[i]['camera_id'] != sightings[i-1]['camera_id']:
                    movements.append({
                        'person_id': person_id,
                        'from_camera': sightings[i-1]['camera_id'],
                        'to_camera': sightings[i]['camera_id'],
                        'from_time': sightings[i-1]['timestamp'],
                        'to_time': sightings[i]['timestamp'],
                        'duration': sightings[i]['timestamp'] - sightings[i-1]['timestamp']
                    })
        
        # Sort by most recent
        movements.sort(key=lambda x: x['to_time'], reverse=True)
        return movements
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the re-identification system
        
        Returns:
            Dictionary of statistics
        """
        camera_counts = defaultdict(int)
        for data in self.people_tracking.values():
            for camera in data['cameras']:
                camera_counts[camera] += 1
        
        now = time.time()
        active_last_hour = sum(1 for data in self.people_tracking.values() 
                              if now - data['last_seen'] <= 3600)
        
        return {
            'total_people': len(self.people_tracking),
            'active_last_hour': active_last_hour,
            'total_sightings': sum(len(data['sightings']) for data in self.people_tracking.values()),
            'cameras': {
                'total': len(camera_counts),
                'people_per_camera': dict(camera_counts)
            },
            'multi_camera_people': sum(1 for data in self.people_tracking.values() 
                                      if len(data['cameras']) > 1)
        }