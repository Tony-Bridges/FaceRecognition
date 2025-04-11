import logging
import random
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)

class ServiceNode:
    """Represents a node in the service cluster"""
    
    def __init__(self, node_id: str, address: str, port: int, service_type: str, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a service node
        
        Args:
            node_id: Unique identifier for the node
            address: Network address/hostname
            port: Port number
            service_type: Type of service (e.g., 'recognition', 'detection', 'api')
            metadata: Additional node metadata
        """
        self.node_id = node_id
        self.address = address
        self.port = port
        self.service_type = service_type
        self.metadata = metadata or {}
        self.status = "unknown"  # "healthy", "unhealthy", "unknown"
        self.last_health_check = 0
        self.health_check_failures = 0
        self.load = 0.0  # Current load (0.0 - 1.0)
        self.last_used = 0  # Timestamp when last used
        self.response_times = []  # Recent response times
        
    def update_status(self, status: str):
        """Update node status"""
        self.status = status
        self.last_health_check = time.time()
        
        if status == "healthy":
            self.health_check_failures = 0
        else:
            self.health_check_failures += 1
    
    def update_load(self, load: float):
        """Update node load"""
        self.load = max(0.0, min(1.0, load))  # Ensure in range 0-1
    
    def add_response_time(self, response_time: float):
        """Add a response time measurement"""
        self.response_times.append(response_time)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
    
    def get_average_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'node_id': self.node_id,
            'address': self.address,
            'port': self.port,
            'status': self.status,
            'service_type': self.service_type,
            'load': self.load,
            'health_check_failures': self.health_check_failures,
            'last_health_check': self.last_health_check,
            'avg_response_time': self.get_average_response_time(),
            'metadata': self.metadata
        }


class LoadBalancer:
    """
    Load balancer for distributing requests across service nodes
    
    Supports multiple strategies:
    - round_robin: Rotate through nodes sequentially
    - least_connections: Send to node with fewest active connections
    - least_load: Send to node with lowest reported load
    - fastest_response: Send to node with fastest average response time
    - weighted_random: Randomly select based on weights
    """
    
    def __init__(self, 
                strategy: str = "round_robin",
                health_check_interval: int = 30,
                node_failure_threshold: int = 3,
                auto_failover: bool = True):
        """
        Initialize the load balancer
        
        Args:
            strategy: Load balancing strategy
            health_check_interval: Seconds between health checks
            node_failure_threshold: Number of failed checks before marking node unhealthy
            auto_failover: Whether to automatically failover to healthy nodes
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.node_failure_threshold = node_failure_threshold
        self.auto_failover = auto_failover
        
        # Node storage
        # service_type -> List[ServiceNode]
        self.nodes: Dict[str, List[ServiceNode]] = defaultdict(list)
        
        # Round robin counters
        # service_type -> current_index
        self.rr_counters: Dict[str, int] = defaultdict(int)
        
        # Active connections
        # node_id -> count
        self.active_connections: Dict[str, int] = defaultdict(int)
        
        # Health check function
        self.health_check_function: Optional[Callable[[ServiceNode], bool]] = None
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Health check thread
        self.health_check_thread = None
        self.stop_health_checks = threading.Event()
        
        logger.info(f"Initialized load balancer with {strategy} strategy")
    
    def register_node(self, node: ServiceNode) -> bool:
        """
        Register a service node
        
        Args:
            node: ServiceNode to register
            
        Returns:
            True if registration was successful
        """
        with self.lock:
            # Check if node with same ID already exists
            for existing_node in self.nodes[node.service_type]:
                if existing_node.node_id == node.node_id:
                    # Update existing node
                    existing_node.address = node.address
                    existing_node.port = node.port
                    existing_node.metadata = node.metadata
                    logger.info(f"Updated existing node {node.node_id}")
                    return True
            
            # Add new node
            self.nodes[node.service_type].append(node)
            logger.info(f"Registered new node {node.node_id} for service {node.service_type}")
            
            # Start health check thread if needed
            self._ensure_health_check_thread()
            
            return True
    
    def deregister_node(self, service_type: str, node_id: str) -> bool:
        """
        Deregister a service node
        
        Args:
            service_type: Service type
            node_id: Node ID to deregister
            
        Returns:
            True if deregistration was successful
        """
        with self.lock:
            if service_type not in self.nodes:
                return False
                
            pre_count = len(self.nodes[service_type])
            self.nodes[service_type] = [
                node for node in self.nodes[service_type] 
                if node.node_id != node_id
            ]
            
            success = len(self.nodes[service_type]) < pre_count
            if success:
                logger.info(f"Deregistered node {node_id} from service {service_type}")
                
                # Clean up active connections
                if node_id in self.active_connections:
                    del self.active_connections[node_id]
            
            return success
    
    def get_node(self, service_type: str) -> Optional[ServiceNode]:
        """
        Get a node for a service based on the load balancing strategy
        
        Args:
            service_type: Service type to get a node for
            
        Returns:
            ServiceNode or None if no healthy nodes available
        """
        with self.lock:
            if service_type not in self.nodes or not self.nodes[service_type]:
                return None
                
            # Filter to healthy nodes if auto_failover is enabled
            available_nodes = self.nodes[service_type]
            if self.auto_failover:
                available_nodes = [
                    node for node in available_nodes 
                    if node.status == "healthy" or (
                        node.status == "unknown" and 
                        node.health_check_failures < self.node_failure_threshold
                    )
                ]
                
            if not available_nodes:
                logger.warning(f"No healthy nodes available for service {service_type}")
                # Fall back to all nodes if none are healthy
                available_nodes = self.nodes[service_type]
                if not available_nodes:
                    return None
            
            # Apply load balancing strategy
            if self.strategy == "round_robin":
                return self._get_round_robin_node(service_type, available_nodes)
                
            elif self.strategy == "least_connections":
                return self._get_least_connections_node(available_nodes)
                
            elif self.strategy == "least_load":
                return self._get_least_load_node(available_nodes)
                
            elif self.strategy == "fastest_response":
                return self._get_fastest_response_node(available_nodes)
                
            elif self.strategy == "weighted_random":
                return self._get_weighted_random_node(available_nodes)
                
            else:
                # Default to round robin
                return self._get_round_robin_node(service_type, available_nodes)
    
    def _get_round_robin_node(self, service_type: str, nodes: List[ServiceNode]) -> ServiceNode:
        """Get node using round robin strategy"""
        idx = self.rr_counters[service_type] % len(nodes)
        self.rr_counters[service_type] = (idx + 1) % len(nodes)
        node = nodes[idx]
        node.last_used = time.time()
        return node
    
    def _get_least_connections_node(self, nodes: List[ServiceNode]) -> ServiceNode:
        """Get node with least active connections"""
        min_connections = float('inf')
        selected_node = None
        
        for node in nodes:
            conn_count = self.active_connections.get(node.node_id, 0)
            if conn_count < min_connections:
                min_connections = conn_count
                selected_node = node
                
        selected_node.last_used = time.time()
        return selected_node
    
    def _get_least_load_node(self, nodes: List[ServiceNode]) -> ServiceNode:
        """Get node with least reported load"""
        min_load = float('inf')
        selected_node = None
        
        for node in nodes:
            if node.load < min_load:
                min_load = node.load
                selected_node = node
                
        selected_node.last_used = time.time()
        return selected_node
    
    def _get_fastest_response_node(self, nodes: List[ServiceNode]) -> ServiceNode:
        """Get node with fastest average response time"""
        min_time = float('inf')
        selected_node = None
        
        for node in nodes:
            avg_time = node.get_average_response_time()
            # If no response time data, use load or defaults to zero
            if avg_time == 0:
                avg_time = node.load * 100  # Convert load to milliseconds scale
                
            if avg_time < min_time:
                min_time = avg_time
                selected_node = node
                
        # If still no node selected (all have zero metrics), use first node
        if not selected_node and nodes:
            selected_node = nodes[0]
                
        selected_node.last_used = time.time()
        return selected_node
    
    def _get_weighted_random_node(self, nodes: List[ServiceNode]) -> ServiceNode:
        """Get node using weighted random selection based on load"""
        # Calculate weights (inverse of load)
        weights = []
        for node in nodes:
            # Ensure load is never 1.0 to avoid zero weight
            weight = 1.0 - min(node.load, 0.99)
            weights.append(weight)
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            # If all weights are zero, use equal weights
            weights = [1.0] * len(nodes)
            total_weight = float(len(nodes))
            
        normalized_weights = [w / total_weight for w in weights]
        
        # Cumulative distribution
        cumulative = [sum(normalized_weights[:i+1]) for i in range(len(normalized_weights))]
        
        # Random selection
        r = random.random()
        for i, threshold in enumerate(cumulative):
            if r <= threshold:
                nodes[i].last_used = time.time()
                return nodes[i]
                
        # Fallback to last node
        nodes[-1].last_used = time.time()
        return nodes[-1]
    
    def mark_connection_started(self, node_id: str):
        """Mark a connection as started on a node"""
        with self.lock:
            self.active_connections[node_id] += 1
    
    def mark_connection_completed(self, node_id: str, response_time: Optional[float] = None):
        """Mark a connection as completed on a node"""
        with self.lock:
            if node_id in self.active_connections:
                self.active_connections[node_id] = max(0, self.active_connections[node_id] - 1)
                
            # Find node and update response time
            if response_time is not None:
                for nodes in self.nodes.values():
                    for node in nodes:
                        if node.node_id == node_id:
                            node.add_response_time(response_time)
                            break
    
    def register_health_check(self, check_function: Callable[[ServiceNode], bool]):
        """
        Register a health check function
        
        Args:
            check_function: Function that takes a ServiceNode and returns True if healthy
        """
        self.health_check_function = check_function
        logger.info("Registered health check function")
    
    def _ensure_health_check_thread(self):
        """Ensure health check thread is running"""
        if (self.health_check_thread is None or 
            not self.health_check_thread.is_alive()) and self.health_check_function:
            self.stop_health_checks.clear()
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self.health_check_thread.start()
            logger.info("Started health check thread")
    
    def _health_check_loop(self):
        """Background thread for health checks"""
        while not self.stop_health_checks.is_set():
            try:
                self._run_health_checks()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                
            # Sleep until next check interval
            for _ in range(self.health_check_interval):
                if self.stop_health_checks.is_set():
                    break
                time.sleep(1)
    
    def _run_health_checks(self):
        """Run health checks on all nodes"""
        if not self.health_check_function:
            return
            
        with self.lock:
            for service_type, nodes in self.nodes.items():
                for node in nodes:
                    try:
                        is_healthy = self.health_check_function(node)
                        node.update_status("healthy" if is_healthy else "unhealthy")
                        if not is_healthy:
                            logger.warning(f"Node {node.node_id} health check failed ({node.health_check_failures} failures)")
                    except Exception as e:
                        logger.error(f"Error checking health of node {node.node_id}: {e}")
                        node.update_status("unknown")
    
    def stop(self):
        """Stop the load balancer and health check thread"""
        self.stop_health_checks.set()
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        logger.info("Stopped load balancer")
    
    def get_service_stats(self, service_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about services and nodes
        
        Args:
            service_type: Optional service type to filter by
            
        Returns:
            Dictionary of statistics
        """
        with self.lock:
            stats = {
                'total_services': len(self.nodes),
                'total_nodes': sum(len(nodes) for nodes in self.nodes.values()),
                'healthy_nodes': sum(
                    sum(1 for node in nodes if node.status == "healthy")
                    for nodes in self.nodes.values()
                ),
                'strategy': self.strategy,
                'services': {}
            }
            
            for svc_type, nodes in self.nodes.items():
                if service_type and svc_type != service_type:
                    continue
                    
                stats['services'][svc_type] = {
                    'node_count': len(nodes),
                    'healthy_nodes': sum(1 for node in nodes if node.status == "healthy"),
                    'unhealthy_nodes': sum(1 for node in nodes if node.status == "unhealthy"),
                    'average_load': sum(node.load for node in nodes) / len(nodes) if nodes else 0,
                    'total_connections': sum(self.active_connections.get(node.node_id, 0) for node in nodes),
                    'nodes': [node.to_dict() for node in nodes]
                }
            
            return stats