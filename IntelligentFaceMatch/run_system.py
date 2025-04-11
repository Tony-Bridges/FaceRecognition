#!/usr/bin/env python3
import logging
import os
import sys
import time
import json
from system_manager import SystemManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_path: str = None):
    """
    Load system configuration from a file or use default
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not config_path or not os.path.exists(config_path):
        logger.info("Using default configuration")
        return {}
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def main():
    """Main entry point"""
    logger.info("Starting face recognition system")
    
    # Load configuration
    config_path = os.environ.get('SYSTEM_CONFIG_PATH')
    config = load_config(config_path)
    
    # Initialize system manager
    system = SystemManager(config)
    
    try:
        # Initialize all components
        if not system.initialize():
            logger.error("Failed to initialize system")
            return 1
            
        logger.info("System initialized successfully")
        
        # In a real application, this would likely start a web server
        # For this demo, we'll just keep the process running
        logger.info("System running, press Ctrl+C to stop")
        
        # Main loop
        while True:
            # Get and log system status periodically
            status = system.get_system_status()
            logger.info(f"System status: initialized={status['initialized']}, "
                       f"components={len(status['components'])}")
            
            # Sleep for a while
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        # Ensure clean shutdown
        system.shutdown()
        logger.info("System shutdown complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())