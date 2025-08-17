"""
Data Grabber - Base class and implementations for acquiring data from various sources
and pushing it to the viewer store.

This module provides a flexible framework for different data acquisition methods
while maintaining a consistent interface to the viewer.
"""

import time
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Hardcoded configuration for demo purposes
# TODO: Move to config file later
CAMERA_HOST = "192.168.1.100"
CAMERA_PORT = 5555
CAMERA_TOPIC = "camera_frames"

# Data processing settings
POLLING_INTERVAL = 0.5  # seconds
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGrabber(ABC):
    """
    Base class for data acquisition from various sources.
    
    This class provides a common interface and polling loop for different
    data acquisition methods (ZMQ, Pulsar, etc.).
    """
    
    def __init__(self, store, polling_interval: float = POLLING_INTERVAL):
        """
        Initialize the data grabber.
        
        Args:
            store: Reference to the viewer's data store
            polling_interval: Time between data acquisition attempts (seconds)
        """
        self.store = store
        self.polling_interval = polling_interval
        self.running = False
        self.thread = None
        self.shot_counter = 0
        
        logger.info(f"DataGrabber initialized with {polling_interval}s polling interval")
    
    @abstractmethod
    def get_data(self) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
        """
        Abstract method to acquire data from the source.
        
        Returns:
            Tuple of (frames, metadata) or None if no data available
            - frames: Dict mapping frame names to 2D numpy arrays
            - metadata: Dict of experimental parameters and metadata
        """
        pass
    
    @abstractmethod
    def _regulate_data(self, frames: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], int, str]:
        """
        Abstract method to regulate and convert camera data format to viewer-compatible format.
        
        This method handles data format conversion, validation, and preprocessing
        to ensure compatibility with the viewer store.
        
        Args:
            frames: Raw frames from camera source
            metadata: Raw metadata from camera source
            
        Returns:
            Tuple of (processed_frames, processed_metadata, shot_id, shot_name)
            - processed_frames: Frames ready for viewer store
            - processed_metadata: Metadata ready for viewer store
            - shot_id: Unique identifier for the shot
            - shot_name: Human-readable name for the shot
        """
        pass
    
    def start(self):
        """Start the data acquisition loop in a separate thread."""
        if self.running:
            logger.warning("DataGrabber is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("DataGrabber started")
    
    def stop(self):
        """Stop the data acquisition loop."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        logger.info("DataGrabber stopped")
    
    def _run_loop(self):
        """Main polling loop that continuously acquires and pushes data."""
        logger.info("Data acquisition loop started")
        
        while self.running:
            try:
                # Attempt to get data from the source
                result = self.get_data()
                
                if result is not None:
                    frames, metadata = result
                    
                    # Process data through regulator before pushing to store
                    processed_frames, processed_metadata, shot_id, shot_name = self._regulate_data(frames, metadata)
                    
                    # Push data to the viewer store
                    actual_shot_id = self.store.add_shot(
                        frames=processed_frames,
                        meta=processed_metadata,
                        shot_name=shot_name
                    )
                    
                    logger.info(f"Pushed shot {actual_shot_id} ({shot_name}) with {len(processed_frames)} frames")
                
                # Wait before next attempt
                time.sleep(self.polling_interval)
                
            except Exception as e:
                logger.error(f"Error in data acquisition loop: {e}")
                time.sleep(self.polling_interval)
        
        logger.info("Data acquisition loop stopped")
    
    def is_running(self) -> bool:
        """Check if the data grabber is currently running."""
        return self.running and self.thread and self.thread.is_alive()


class RemoteInstrumentImageGrabber(DataGrabber):
    """
    Data grabber using quetzal-instruments RemoteInstrument for camera control.
    
    This implementation connects to a remote camera server and acquires image data
    with duplicate detection using image hashing for robust shot identification.
    """
    
    def __init__(self, store, host: str, port: int, **kwargs):
        """
        Initialize the remote instrument image grabber.
        
        Args:
            store: Reference to the viewer's data store
            host: Camera server host address
            port: Camera server port
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(store, **kwargs)
        self.host = host
        self.port = port
        
        # Initialize RemoteInstrument - fail fast if connection fails
        try:
            from quetzal.instruments.remote import RemoteInstrument
            self.remote_instrument = RemoteInstrument(host, port)
            logger.info(f"Connected to camera server at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to camera server: {e}")
            import sys
            sys.exit(1)  # Quit app on connection failure
        
        # Initialize duplicate detection
        self._last_hash = None
        self._last_timestamp = None
    
    def get_data(self) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
        """
        Acquire data from remote camera server.
        
        Returns:
            Tuple of (frames, metadata) or None if no data available
        """
        try:
            # Call remote method and extract data
            result = self.remote_instrument.get_shot_data()
            frames = result.get('frames', {})
            metadata = result.get('meta', {})
            
            # Check if this is a new shot
            if not self._is_new_shot(frames, metadata):
                return None  # Skip duplicate shots

            return frames, metadata
            
        except Exception as e:
            logger.error(f"Error getting shot data: {e}")
            return None
    
    def _is_new_shot(self, frames: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> bool:
        """
        Check if this is a new shot based on image content hashing.
        
        Args:
            frames: Dictionary of frame data
            metadata: Shot metadata
            
        Returns:
            True if this is a new shot, False if duplicate
        """
        try:
            # Get the main frame for hashing (usually 'raw' or 'processed')
            main_frame = frames.get('raw') or frames.get('processed') or next(iter(frames.values()))
            if main_frame is None:
                return True
            
            # Ensure frame is numpy array
            if not isinstance(main_frame, np.ndarray):
                main_frame = np.asarray(main_frame)
            
            # Hash the frame content
            import hashlib
            content_hash = hashlib.blake2b(main_frame.tobytes(), digest_size=16)
            current_hash = content_hash.hexdigest()
            
            # Check if hash is different from last shot
            if self._last_hash is not None:
                if current_hash == self._last_hash:
                    return False  # Duplicate shot
            
            # Update hash for next comparison
            self._last_hash = current_hash
            return True
            
        except Exception as e:
            logger.warning(f"Error in duplicate detection: {e}")
            # If hashing fails, assume it's new data
            return True
    
    def _regulate_data(self, frames: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], int, str]:
        """
        Regulate remote instrument data format for viewer compatibility.
        
        Args:
            frames: Raw frames from remote camera source
            metadata: Raw metadata from remote camera source
            
        Returns:
            Tuple of (processed_frames, processed_metadata, shot_id, shot_name)
        """
        # Extract shot info from metadata when possible
        shot_id = metadata.get('shot_id')
        shot_name = metadata.get('shot_name')
        
        # Generate shot info if not provided in metadata
        if shot_id is None:
            self.shot_counter += 1
            shot_id = self.shot_counter
        
        if shot_name is None:
            # shot_name = f"Remote_{shot_id}"
            shot_name = metadata.get('timestamp')   
        
        # Minimal processing - just ensure frames are numpy arrays
        processed_frames = {}
        for frame_name, frame_data in frames.items():
            if frame_data is not None:
                # Convert to numpy array if needed
                if not isinstance(frame_data, np.ndarray):
                    frame_data = np.asarray(frame_data)
                processed_frames[frame_name] = frame_data
        
        # Add timestamp if not present
        if 'timestamp' not in metadata:
            metadata['timestamp'] = time.time()
        
        return processed_frames, metadata, shot_id, shot_name
    
    def stop(self):
        """Stop the data grabber and cleanup remote instrument connection."""
        super().stop()
        try:
            if hasattr(self, 'remote_instrument'):
                self.remote_instrument.close()
                logger.info("Remote instrument connection closed")
        except Exception as e:
            logger.warning(f"Error closing remote instrument: {e}")


# Convenience function for creating grabbers
def create_grabber(grabber_type: str, store, **kwargs) -> DataGrabber:
    """
    Factory function to create data grabbers.
    
    Args:
        grabber_type: Type of grabber ('zmq', 'dummy', 'quetzal', etc.)
        store: Reference to the viewer's data store
        **kwargs: Additional arguments for the specific grabber type
    
    Returns:
        Configured DataGrabber instance
    """
    if grabber_type.lower() == 'zmq':
        return ZMQDataGrabber(store, **kwargs)
    elif grabber_type.lower() == 'dummy':
        return DummyGrabber(store, **kwargs)
    elif grabber_type.lower() == 'quetzal':
        return RemoteInstrumentImageGrabber(store, **kwargs)
    # TODO: Add other grabber types here
    # elif grabber_type.lower() == 'pulsar':
    #     return PulsarDataGrabber(store, **kwargs)
    else:
        raise ValueError(f"Unknown grabber type: {grabber_type}")
