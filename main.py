#!/usr/bin/env python3
"""
Main entry point for the Cold Atom Imaging Dashboard with Data Grabber.

This script starts both the Dash viewer application and the data grabber service,
coordinating their startup and shutdown.
"""

import os
import sys
import time
import signal
import logging
import threading
from typing import Optional

# Add the current directory to Python path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the viewer app and data grabber
from app import app, store
from data_grabber import create_grabber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for service management
viewer_thread: Optional[threading.Thread] = None
data_grabber = None
shutdown_event = threading.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_event.set()


def start_viewer():
    """Start the Dash viewer application in a separate thread."""
    global viewer_thread
    
    def run_viewer():
        try:
            logger.info("Starting Dash viewer application...")
            # Run the app in debug mode for development
            # In production, you might want to use a WSGI server like gunicorn
            app.run(
                debug=False,  # Set to False for production
                host="0.0.0.0",  # Listen on all interfaces
                port=8050,
                use_reloader=False  # Disable reloader when running in thread
            )
        except Exception as e:
            logger.error(f"Error in viewer thread: {e}")
    
    viewer_thread = threading.Thread(target=run_viewer, daemon=True)
    viewer_thread.start()
    logger.info("Dash viewer started in background thread")


def start_data_grabber():
    """Start the data grabber service."""
    global data_grabber
    
    try:
        # Create and start the data grabber
        # You can modify these parameters or read from environment variables
        data_grabber = create_grabber(
            grabber_type='zmq',
            store=store,
            host=os.getenv('CAMERA_HOST', '192.168.1.100'),
            port=int(os.getenv('CAMERA_PORT', '5555')),
            topic=os.getenv('CAMERA_TOPIC', 'camera_frames'),
            polling_interval=float(os.getenv('POLLING_INTERVAL', '0.1'))
        )
        
        logger.info("Starting data grabber...")
        data_grabber.start()
        logger.info("Data grabber started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start data grabber: {e}")
        raise


def stop_services():
    """Stop all running services gracefully."""
    global data_grabber, viewer_thread
    
    logger.info("Stopping services...")
    
    # Stop data grabber
    if data_grabber and data_grabber.is_running():
        logger.info("Stopping data grabber...")
        data_grabber.stop()
        logger.info("Data grabber stopped")
    
    # Note: Dash app runs in a daemon thread, so it will terminate automatically
    # when the main thread exits
    logger.info("Services stopped")


def main():
    """Main function to start and manage all services."""
    logger.info("Starting Cold Atom Imaging Dashboard with Data Grabber...")
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the data grabber first
        start_data_grabber()
        
        # Give the data grabber a moment to initialize
        time.sleep(1)
        
        # Start the viewer application
        start_viewer()
        
        logger.info("All services started successfully!")
        logger.info("Dashboard available at: http://localhost:8050")
        logger.info("Press Ctrl+C to stop all services")
        
        # Wait for shutdown signal
        while not shutdown_event.is_set():
            time.sleep(0.1)
            
            # Check if data grabber is still running
            if data_grabber and not data_grabber.is_running():
                logger.warning("Data grabber stopped unexpectedly")
                break
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup
        stop_services()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
