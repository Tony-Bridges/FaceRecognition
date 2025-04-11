import logging
import time
import threading
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

class ActiveLearningSystem:
    """
    Active learning system to continuously improve face recognition models
    by identifying challenging examples and incorporating human feedback
    """
    
    def __init__(self, 
                data_dir: str = './active_learning_data',
                confidence_threshold: float = 0.65,
                selection_strategy: str = 'least_confidence',
                batch_size: int = 10,
                max_queue_size: int = 100,
                feedback_interval: int = 3600,  # seconds
                auto_retraining: bool = False):
        """
        Initialize the active learning system
        
        Args:
            data_dir: Directory to store active learning data
            confidence_threshold: Threshold below which to consider samples for active learning
            selection_strategy: Strategy for selecting samples ('least_confidence', 'entropy', 'margin')
            batch_size: Number of samples to select in each batch
            max_queue_size: Maximum size of the uncertainty queue
            feedback_interval: How often to request feedback (in seconds)
            auto_retraining: Whether to automatically retrain models with new data
        """
        self.data_dir = data_dir
        self.confidence_threshold = confidence_threshold
        self.selection_strategy = selection_strategy
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.feedback_interval = feedback_interval
        self.auto_retraining = auto_retraining
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'uncertain_samples'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'human_verified'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'model_versions'), exist_ok=True)
        
        # Queue for uncertain samples
        # list of {face_id, embedding, image_path, metadata, timestamp, confidence_scores}
        self.uncertainty_queue = []
        
        # Samples that have received human feedback
        # face_id -> {embedding, verified_name, human_confidence, timestamp, original_metadata}
        self.verified_samples = {}
        
        # Latest model version info
        self.current_model_version = 1
        self.last_training_time = 0
        self.model_performance = {}
        
        # Thread for periodic batch selection
        self.selection_thread = None
        self.stop_thread = threading.Event()
        
        # Callbacks
        self.feedback_callback = None  # Called when feedback is needed
        self.retraining_callback = None  # Called when model retraining is needed
        
        logger.info(f"Initialized active learning system with {selection_strategy} strategy")
        
        # Load any existing data
        self._load_verified_samples()
        
        # Start selection thread if auto mode enabled
        if self.auto_retraining:
            self._start_selection_thread()
    
    def _start_selection_thread(self):
        """Start the background thread for periodic sample selection"""
        if self.selection_thread and self.selection_thread.is_alive():
            return
            
        self.stop_thread.clear()
        self.selection_thread = threading.Thread(
            target=self._periodic_selection_loop,
            daemon=True
        )
        self.selection_thread.start()
        logger.info("Started active learning selection thread")
    
    def _periodic_selection_loop(self):
        """Background loop to periodically select samples and request feedback"""
        while not self.stop_thread.is_set():
            try:
                # Select a batch of samples if we have enough in the queue
                if (len(self.uncertainty_queue) >= self.batch_size and 
                    time.time() - self.last_training_time >= self.feedback_interval):
                    self._select_batch_for_feedback()
                
                # Sleep for a while before checking again
                for _ in range(60):  # Check stop flag more frequently
                    if self.stop_thread.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in active learning selection thread: {e}")
                time.sleep(60)  # Sleep and retry
    
    def stop(self):
        """Stop the background thread"""
        self.stop_thread.set()
        if self.selection_thread:
            self.selection_thread.join(timeout=5)
        logger.info("Stopped active learning system")
    
    def add_uncertain_sample(self, face_id: str, embedding: np.ndarray, 
                            image_path: str, metadata: Dict,
                            confidence_scores: Dict[str, float]) -> bool:
        """
        Add a sample to the uncertainty queue for potential active learning
        
        Args:
            face_id: Unique ID of the face
            embedding: Face embedding vector
            image_path: Path to the face image
            metadata: Additional metadata (name, camera_id, etc.)
            confidence_scores: Dictionary mapping potential names to confidence scores
            
        Returns:
            True if added to queue, False otherwise
        """
        # Check if this face is already verified
        if face_id in self.verified_samples:
            return False
            
        # Check if confidence is below threshold
        if self._calculate_uncertainty(confidence_scores) < self.confidence_threshold:
            return False
            
        # Add to queue
        self.uncertainty_queue.append({
            'face_id': face_id,
            'embedding': embedding,
            'image_path': image_path,
            'metadata': metadata,
            'timestamp': time.time(),
            'confidence_scores': confidence_scores
        })
        
        # Trim queue if needed
        if len(self.uncertainty_queue) > self.max_queue_size:
            self.uncertainty_queue.sort(
                key=lambda x: self._calculate_uncertainty(x['confidence_scores']),
                reverse=True
            )
            self.uncertainty_queue = self.uncertainty_queue[:self.max_queue_size]
        
        logger.debug(f"Added face {face_id} to uncertainty queue (queue size: {len(self.uncertainty_queue)})")
        return True
    
    def _calculate_uncertainty(self, confidence_scores: Dict[str, float]) -> float:
        """
        Calculate uncertainty based on the selected strategy
        
        Args:
            confidence_scores: Dictionary mapping class names to confidence scores
            
        Returns:
            Uncertainty score (higher = more uncertain)
        """
        if not confidence_scores:
            return 1.0
            
        if self.selection_strategy == 'least_confidence':
            # 1 - maximum confidence
            return 1.0 - max(confidence_scores.values())
            
        elif self.selection_strategy == 'margin':
            # Small difference between top two predictions indicates uncertainty
            sorted_confidences = sorted(confidence_scores.values(), reverse=True)
            if len(sorted_confidences) >= 2:
                return 1.0 - (sorted_confidences[0] - sorted_confidences[1])
            return 0.0
            
        elif self.selection_strategy == 'entropy':
            # Entropy of the probability distribution
            probs = list(confidence_scores.values())
            probs_sum = sum(probs)
            if probs_sum == 0:
                return 1.0
            # Normalize
            probs = [p/probs_sum for p in probs]
            # Calculate entropy
            entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs)
            # Normalize to [0, 1]
            max_entropy = np.log(len(probs)) if len(probs) > 0 else 1
            return entropy / max_entropy if max_entropy > 0 else 0.0
            
        # Default to least confidence
        return 1.0 - max(confidence_scores.values())
    
    def _select_batch_for_feedback(self) -> List[Dict]:
        """
        Select a batch of samples for human feedback
        
        Returns:
            List of samples requiring feedback
        """
        if not self.uncertainty_queue:
            return []
            
        # Sort by uncertainty (highest first)
        self.uncertainty_queue.sort(
            key=lambda x: self._calculate_uncertainty(x['confidence_scores']),
            reverse=True
        )
        
        # Select top N samples
        selected_batch = self.uncertainty_queue[:self.batch_size]
        
        # Remove selected samples from queue
        self.uncertainty_queue = self.uncertainty_queue[self.batch_size:]
        
        logger.info(f"Selected {len(selected_batch)} samples for human feedback")
        
        # Call feedback callback if registered
        if self.feedback_callback:
            self.feedback_callback(selected_batch)
        
        return selected_batch
    
    def register_feedback_callback(self, callback: Callable[[List[Dict]], None]):
        """
        Register a callback to be called when human feedback is needed
        
        Args:
            callback: Function to call with the list of samples needing feedback
        """
        self.feedback_callback = callback
        logger.info("Registered feedback callback")
    
    def register_retraining_callback(self, callback: Callable[[Dict], None]):
        """
        Register a callback to be called when model retraining is needed
        
        Args:
            callback: Function to call with training parameters
        """
        self.retraining_callback = callback
        logger.info("Registered retraining callback")
    
    def add_human_feedback(self, face_id: str, verified_name: str, 
                          human_confidence: float = 1.0) -> bool:
        """
        Add human verification for a face
        
        Args:
            face_id: Face identifier
            verified_name: Human-verified correct name
            human_confidence: Confidence in the human verification (0.0-1.0)
            
        Returns:
            True if feedback was added successfully
        """
        # Find the sample in our uncertainty queue
        sample = None
        for s in self.uncertainty_queue:
            if s['face_id'] == face_id:
                sample = s
                break
        
        if not sample:
            logger.warning(f"Sample {face_id} not found in uncertainty queue")
            return False
        
        # Add to verified samples
        self.verified_samples[face_id] = {
            'embedding': sample['embedding'],
            'verified_name': verified_name,
            'human_confidence': human_confidence,
            'timestamp': time.time(),
            'original_metadata': sample['metadata'],
            'image_path': sample['image_path']
        }
        
        # Remove from uncertainty queue if still there
        self.uncertainty_queue = [s for s in self.uncertainty_queue if s['face_id'] != face_id]
        
        # Save to disk
        self._save_verified_sample(face_id)
        
        logger.info(f"Added human feedback for face {face_id}: {verified_name}")
        
        # Check if we need to retrain
        self._check_retraining_needed()
        
        return True
    
    def _save_verified_sample(self, face_id: str):
        """Save a verified sample to disk"""
        if face_id not in self.verified_samples:
            return
            
        sample = self.verified_samples[face_id]
        
        # Save metadata
        metadata_path = os.path.join(
            self.data_dir, 'human_verified', f"{face_id}.json"
        )
        
        # Convert numpy array to list for JSON serialization
        sample_json = {k: v for k, v in sample.items()}
        if 'embedding' in sample_json:
            sample_json['embedding'] = sample_json['embedding'].tolist()
            
        with open(metadata_path, 'w') as f:
            json.dump(sample_json, f)
    
    def _load_verified_samples(self):
        """Load verified samples from disk"""
        verified_dir = os.path.join(self.data_dir, 'human_verified')
        if not os.path.exists(verified_dir):
            return
            
        for filename in os.listdir(verified_dir):
            if not filename.endswith('.json'):
                continue
                
            try:
                filepath = os.path.join(verified_dir, filename)
                with open(filepath, 'r') as f:
                    sample = json.load(f)
                
                face_id = os.path.splitext(filename)[0]
                
                # Convert embedding back to numpy array
                if 'embedding' in sample and isinstance(sample['embedding'], list):
                    sample['embedding'] = np.array(sample['embedding'], dtype=np.float32)
                    
                self.verified_samples[face_id] = sample
            except Exception as e:
                logger.error(f"Error loading verified sample {filename}: {e}")
        
        logger.info(f"Loaded {len(self.verified_samples)} verified samples")
    
    def _check_retraining_needed(self):
        """Check if model retraining is needed based on new samples"""
        # Simple heuristic: Retrain if we have enough new samples since last training
        samples_since_training = sum(
            1 for s in self.verified_samples.values()
            if s['timestamp'] > self.last_training_time
        )
        
        if samples_since_training >= self.batch_size and self.auto_retraining:
            self._trigger_retraining()
    
    def _trigger_retraining(self):
        """Trigger model retraining"""
        # Prepare training data
        training_data = {
            'samples': [
                {
                    'face_id': face_id,
                    'name': sample['verified_name'],
                    'embedding': sample['embedding'].tolist(),
                    'confidence': sample['human_confidence'],
                    'metadata': sample['original_metadata']
                }
                for face_id, sample in self.verified_samples.items()
            ],
            'model_version': self.current_model_version + 1,
            'timestamp': time.time()
        }
        
        # Call retraining callback if registered
        if self.retraining_callback:
            self.retraining_callback(training_data)
            self.last_training_time = time.time()
            self.current_model_version += 1
            logger.info(f"Triggered model retraining (version {self.current_model_version})")
    
    def update_model_performance(self, metrics: Dict[str, float]):
        """
        Update performance metrics for the current model
        
        Args:
            metrics: Dictionary with performance metrics
        """
        self.model_performance = {
            'version': self.current_model_version,
            'timestamp': time.time(),
            'metrics': metrics
        }
        
        # Save model performance
        performance_path = os.path.join(
            self.data_dir, 'model_versions', f"model_v{self.current_model_version}_performance.json"
        )
        
        with open(performance_path, 'w') as f:
            json.dump(self.model_performance, f)
            
        logger.info(f"Updated performance metrics for model v{self.current_model_version}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the active learning system
        
        Returns:
            Dictionary with statistics
        """
        return {
            'uncertainty_queue_size': len(self.uncertainty_queue),
            'verified_samples': len(self.verified_samples),
            'current_model_version': self.current_model_version,
            'last_training_time': self.last_training_time,
            'time_since_training': time.time() - self.last_training_time,
            'model_performance': self.model_performance.get('metrics', {}),
            'feedback_interval': self.feedback_interval,
            'auto_retraining': self.auto_retraining,
            'selection_strategy': self.selection_strategy
        }