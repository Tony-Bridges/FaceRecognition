import logging
import os
from typing import Dict, List, Any, Optional, Callable
import threading
import time

# Import advanced features
from advanced_features import (
    EmotionDetector,
    AgeGenderEstimator,
    PersonReIdentification,
    ActiveLearningSystem
)

# Import scaling components
from scaling import LoadBalancer, ServiceNode
from scaling.sharding import ShardManager, UniformShardingStrategy

# Import monitoring
from monitoring import MetricsCollector

logger = logging.getLogger(__name__)

class SystemManager:
    """
    Manages the entire face recognition system, integrating all components
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the system manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.initialized = False
        self.components = {}
        self.metrics = None
        self.system_lock = threading.RLock()
        
        # Background tasks
        self.background_threads = []
        self.stop_background_tasks = threading.Event()
        
        # Setup default configuration
        self._setup_default_config()
        
        logger.info("System manager initialized")
    
    def _setup_default_config(self):
        """Setup default configuration if not provided"""
        if 'metrics' not in self.config:
            self.config['metrics'] = {
                'enabled': True,
                'port': 9090
            }
            
        if 'scaling' not in self.config:
            self.config['scaling'] = {
                'load_balancing': {
                    'strategy': 'round_robin',
                    'health_check_interval': 30
                },
                'sharding': {
                    'enabled': False,
                    'strategy': 'uniform',
                    'shard_count': 1
                }
            }
            
        if 'features' not in self.config:
            self.config['features'] = {
                'emotion_detection': False,
                'age_gender_estimation': False,
                're_identification': True,
                'active_learning': False
            }
    
    def initialize(self) -> bool:
        """
        Initialize all system components
        
        Returns:
            True if initialization was successful
        """
        with self.system_lock:
            if self.initialized:
                return True
                
            try:
                # Initialize metrics first for other components to use
                self._init_metrics()
                
                # Initialize core components
                self._init_database()
                self._init_recognition_service()
                
                # Initialize scaling components
                self._init_load_balancer()
                self._init_sharding()
                
                # Initialize advanced features
                self._init_advanced_features()
                
                # Start background tasks
                self._start_background_tasks()
                
                self.initialized = True
                logger.info("System manager successfully initialized all components")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize system: {e}")
                # Try to clean up any partially initialized components
                self.shutdown()
                return False
    
    def _init_metrics(self):
        """Initialize metrics collection"""
        if not self.config['metrics']['enabled']:
            logger.info("Metrics collection disabled")
            return
            
        try:
            self.metrics = MetricsCollector(port=self.config['metrics']['port'])
            self.components['metrics'] = self.metrics
            logger.info(f"Metrics collection initialized on port {self.config['metrics']['port']}")
        except Exception as e:
            logger.warning(f"Failed to initialize metrics: {e}")
    
    def _init_database(self):
        """Initialize database connections"""
        # This would typically connect to your face database
        # For now, we're just creating a placeholder
        self.components['database'] = {
            'initialized': True,
            'type': 'postgres'
        }
        logger.info("Database initialized")
    
    def _init_recognition_service(self):
        """Initialize face recognition service"""
        # This would typically initialize your face recognition models
        # For now, we're just creating a placeholder
        self.components['recognition'] = {
            'initialized': True,
            'model': 'insightface'  # placeholder
        }
        logger.info("Recognition service initialized")
    
    def _init_load_balancer(self):
        """Initialize load balancing if configured"""
        scaling_config = self.config['scaling']
        if not scaling_config.get('load_balancing', {}).get('enabled', False):
            logger.info("Load balancing disabled")
            return
            
        lb_config = scaling_config['load_balancing']
        lb = LoadBalancer(
            strategy=lb_config.get('strategy', 'round_robin'),
            health_check_interval=lb_config.get('health_check_interval', 30),
            node_failure_threshold=lb_config.get('node_failure_threshold', 3),
            auto_failover=lb_config.get('auto_failover', True)
        )
        
        # Register health check
        lb.register_health_check(self._health_check_node)
        
        self.components['load_balancer'] = lb
        logger.info(f"Load balancer initialized with {lb.strategy} strategy")
    
    def _init_sharding(self):
        """Initialize database sharding if configured"""
        scaling_config = self.config['scaling']
        if not scaling_config.get('sharding', {}).get('enabled', False):
            logger.info("Database sharding disabled")
            return
            
        shard_config = scaling_config['sharding']
        strategy_name = shard_config.get('strategy', 'uniform')
        
        # Select sharding strategy
        if strategy_name == 'uniform':
            strategy = UniformShardingStrategy()
        elif strategy_name == 'geographic':
            region_mapping = shard_config.get('region_mapping', {})
            strategy = GeographicShardingStrategy(region_mapping)
        elif strategy_name == 'camera_group':
            camera_mapping = shard_config.get('camera_mapping', {})
            strategy = CameraGroupShardingStrategy(camera_mapping)
        else:
            strategy = UniformShardingStrategy()
            
        # Get shard URLs (in a real system, these would be configured or discovered)
        shard_urls = shard_config.get('shard_urls', [os.environ.get('DATABASE_URL', '')])
        
        # Create shard manager
        shard_manager = ShardManager(
            shard_urls=shard_urls,
            strategy=strategy,
            config=shard_config
        )
        
        self.components['shard_manager'] = shard_manager
        logger.info(f"Shard manager initialized with {strategy_name} strategy and {len(shard_urls)} shards")
    
    def _init_advanced_features(self):
        """Initialize advanced features based on configuration"""
        features_config = self.config['features']
        
        # Emotion detection
        if features_config.get('emotion_detection', False):
            self.components['emotion_detector'] = EmotionDetector(
                backend=features_config.get('emotion_detection_backend', 'opencv'),
                model_path=features_config.get('emotion_model_path')
            )
            logger.info("Emotion detection initialized")
            
        # Age and gender estimation
        if features_config.get('age_gender_estimation', False):
            self.components['age_gender_estimator'] = AgeGenderEstimator(
                backend=features_config.get('age_gender_backend', 'opencv'),
                age_model_path=features_config.get('age_model_path'),
                gender_model_path=features_config.get('gender_model_path')
            )
            logger.info("Age and gender estimation initialized")
            
        # Person re-identification
        if features_config.get('re_identification', False):
            re_id_config = features_config.get('re_identification_config', {})
            self.components['re_identification'] = PersonReIdentification(
                similarity_threshold=re_id_config.get('similarity_threshold', 0.7),
                tracking_timeout=re_id_config.get('tracking_timeout', 300),
                max_history=re_id_config.get('max_history', 1000)
            )
            logger.info("Person re-identification initialized")
            
        # Active learning
        if features_config.get('active_learning', False):
            al_config = features_config.get('active_learning_config', {})
            active_learning = ActiveLearningSystem(
                data_dir=al_config.get('data_dir', './active_learning_data'),
                confidence_threshold=al_config.get('confidence_threshold', 0.65),
                selection_strategy=al_config.get('selection_strategy', 'least_confidence'),
                batch_size=al_config.get('batch_size', 10),
                feedback_interval=al_config.get('feedback_interval', 3600),
                auto_retraining=al_config.get('auto_retraining', False)
            )
            
            # Register feedback callback
            active_learning.register_feedback_callback(self._handle_active_learning_feedback)
            
            # Register retraining callback
            active_learning.register_retraining_callback(self._handle_model_retraining)
            
            self.components['active_learning'] = active_learning
            logger.info("Active learning system initialized")
    
    def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        # System stats monitoring thread
        if self.metrics:
            stats_thread = threading.Thread(
                target=self._system_stats_loop,
                daemon=True
            )
            self.background_threads.append(stats_thread)
            stats_thread.start()
            logger.info("Started system stats monitoring thread")
    
    def _system_stats_loop(self):
        """Background thread for collecting system statistics"""
        while not self.stop_background_tasks.is_set():
            try:
                # Collect system statistics
                # This is a placeholder - in a real system you would use psutil or similar
                memory_usage = 100000000  # 100 MB
                cpu_usage = 5.0  # 5%
                
                # Update metrics
                if self.metrics:
                    self.metrics.update_system_metrics(memory_usage, cpu_usage)
                    
                # Update component-specific metrics
                self._update_component_metrics()
                    
            except Exception as e:
                logger.error(f"Error in system stats loop: {e}")
                
            # Sleep for 10 seconds
            for _ in range(10):
                if self.stop_background_tasks.is_set():
                    break
                time.sleep(1)
    
    def _update_component_metrics(self):
        """Update metrics for all components"""
        if not self.metrics:
            return
            
        # Update database stats
        if 'shard_manager' in self.components:
            shard_manager = self.components['shard_manager']
            stats = shard_manager.get_statistics()
            
            for shard_id, shard_stats in stats.get('per_shard_stats', {}).items():
                face_count = shard_stats.get('face_count', 0)
                self.metrics.update_database_size(shard_id, face_count)
                
        # Update load balancer stats
        if 'load_balancer' in self.components:
            lb = self.components['load_balancer']
            stats = lb.get_service_stats()
            
            # Update active services metrics (if we had prometheus gauges for them)
            # This would track number of active nodes per service
    
    def _health_check_node(self, node: ServiceNode) -> bool:
        """Health check for a service node"""
        # This is a placeholder implementation
        # In a real system, you would ping the node or check its health endpoint
        try:
            # Simulate health check
            return True  # Always healthy for demo
        except Exception:
            return False
    
    def _handle_active_learning_feedback(self, samples: List[Dict]):
        """
        Handle feedback requests from active learning system
        
        Args:
            samples: List of samples requiring feedback
        """
        logger.info(f"Active learning system requesting feedback for {len(samples)} samples")
        # In a real system, this would send the samples to a human reviewer
        # or add them to a queue for review
    
    def _handle_model_retraining(self, training_data: Dict):
        """
        Handle model retraining request from active learning system
        
        Args:
            training_data: Training data and parameters
        """
        logger.info(f"Retraining face recognition model with {len(training_data['samples'])} samples")
        # In a real system, this would trigger model retraining
        # and update the face recognition service with the new model
    
    def get_component(self, name: str) -> Any:
        """
        Get a system component by name
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None if not found
        """
        return self.components.get(name)
    
    def shutdown(self) -> bool:
        """
        Shutdown all system components
        
        Returns:
            True if shutdown was successful
        """
        with self.system_lock:
            if not self.initialized:
                return True
                
            logger.info("Shutting down system components...")
            
            # Stop background tasks
            self.stop_background_tasks.set()
            for thread in self.background_threads:
                thread.join(timeout=5)
            
            # Shutdown active learning
            if 'active_learning' in self.components:
                self.components['active_learning'].stop()
                
            # Shutdown load balancer
            if 'load_balancer' in self.components:
                self.components['load_balancer'].stop()
                
            # Clear all components
            self.components.clear()
            self.initialized = False
            
            logger.info("System components shutdown complete")
            return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status report
        
        Returns:
            Dictionary with system status information
        """
        status = {
            'initialized': self.initialized,
            'components': {},
            'metrics': {}
        }
        
        # Add component status
        for name, component in self.components.items():
            if name == 'metrics':
                continue
                
            # Use component-specific status method if available
            if hasattr(component, 'get_statistics'):
                status['components'][name] = component.get_statistics()
            else:
                status['components'][name] = {'status': 'active'}
        
        # Get sharding status if available
        if 'shard_manager' in self.components:
            status['sharding'] = self.components['shard_manager'].get_statistics()
            
        # Get load balancer status if available
        if 'load_balancer' in self.components:
            status['load_balancing'] = self.components['load_balancer'].get_service_stats()
        
        return status