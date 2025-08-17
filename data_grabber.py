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
POLLING_INTERVAL = 0.1  # seconds
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


class ZMQDataGrabber(DataGrabber):
    """
    ZMQ-based data grabber for receiving camera frames from a ZMQ server.
    
    This implementation connects to a ZMQ publisher/subscriber pattern
    to receive camera data and metadata.
    """
    
    def __init__(self, store, host: str = CAMERA_HOST, port: int = CAMERA_PORT, 
                 topic: str = CAMERA_TOPIC, polling_interval: float = POLLING_INTERVAL):
        """
        Initialize the ZMQ data grabber.
        
        Args:
            store: Reference to the viewer's data store
            host: ZMQ server host address
            port: ZMQ server port
            topic: ZMQ topic to subscribe to
            polling_interval: Time between data acquisition attempts (seconds)
        """
        super().__init__(store, polling_interval)
        self.host = host
        self.port = port
        self.topic = topic
        
        # TODO: Initialize ZMQ connection here
        # - Create ZMQ context
        # - Create subscriber socket
        # - Connect to server
        # - Subscribe to topic
        
        logger.info(f"ZMQDataGrabber initialized for {host}:{port}/{topic}")
    
    def get_data(self) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
        """
        Receive data from ZMQ server.
        
        Returns:
            Tuple of (frames, metadata) or None if no data available
        """
        # TODO: Implement ZMQ data reception
        # - Receive message from ZMQ socket
        # - Parse frame data and metadata
        # - Convert to appropriate numpy arrays
        # - Return (frames, metadata) tuple
        
        # Note: The base class will automatically regulate this data
        # through the _regulate_data method before pushing to the store
        
        # Placeholder return - replace with actual implementation
        return None
    
    def _regulate_data(self, frames: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], int, str]:
        """
        Regulate and convert ZMQ camera data format to viewer-compatible format.
        
        Args:
            frames: Raw frames from ZMQ camera source
            metadata: Raw metadata from ZMQ camera source
            
        Returns:
            Tuple of (processed_frames, processed_metadata, shot_id, shot_name)
        """
        # TODO: Implement ZMQ-specific data regulation
        # - Process frames according to ZMQ camera format
        # - Extract and validate metadata
        # - Generate appropriate shot_id and shot_name
        # - Return processed data ready for viewer store
        
        # Placeholder implementation - replace with actual ZMQ data processing
        processed_frames = {}
        processed_metadata = metadata.copy() if isinstance(metadata, dict) else {}
        
        # Process each frame
        for frame_name, frame_data in frames.items():
            if frame_data is None:
                continue
                
            # Ensure frame is a numpy array
            if not isinstance(frame_data, np.ndarray):
                frame_data = np.array(frame_data)
            
            # Validate frame dimensions
            if frame_data.ndim != 2:
                logger.warning(f"Frame {frame_name} has {frame_data.ndim} dimensions, expected 2D")
                continue
            
            # Convert to float32 for viewer compatibility
            if frame_data.dtype != np.float32:
                frame_data = frame_data.astype(np.float32, copy=False)
            
            # Store processed frame
            processed_frames[frame_name] = frame_data
        
        # Generate shot information
        self.shot_counter += 1
        shot_id = self.shot_counter
        shot_name = f"ZMQ_{shot_id}"
        
        return processed_frames, processed_metadata, shot_id, shot_name
    
    def _cleanup(self):
        """Clean up ZMQ resources."""
        # TODO: Implement ZMQ cleanup
        # - Close socket
        # - Terminate context
        pass
    
    def stop(self):
        """Stop the data grabber and cleanup ZMQ resources."""
        super().stop()
        self._cleanup()


class DummyGrabber(DataGrabber):
    """
    Dummy data grabber for testing and demonstration purposes.
    
    Generates fake 512x512 random images at 1Hz rate with simulated metadata.
    """
    
    def __init__(self, store, image_size: int = 512, frame_rate: float = 1.0, **kwargs):
        """
        Initialize the dummy data grabber.
        
        Args:
            store: Reference to the viewer's data store
            image_size: Size of generated images (width=height)
            frame_rate: Rate of data generation in Hz
            **kwargs: Additional arguments
        """
        super().__init__(store, polling_interval=1.0/frame_rate)
        self.image_size = image_size
        self.frame_rate = frame_rate
        self.frame_counter = 0
        
        logger.info(f"DummyGrabber initialized: {image_size}x{image_size} images at {frame_rate}Hz")
    
    def get_data(self) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
        """
        Generate fake data for testing.
        
        Returns:
            Tuple of (frames, metadata) with simulated camera data
        """
        # Generate random image data
        raw_frame = np.random.normal(1000, 100, size=(self.image_size, self.image_size)).astype(np.float32)
        processed_frame = np.random.normal(500, 50, size=(self.image_size, self.image_size)).astype(np.float32)
        
        # Add some simulated structure to make images more interesting
        x, y = np.meshgrid(np.linspace(-1, 1, self.image_size), np.linspace(-1, 1, self.image_size))
        structure = 200 * np.exp(-(x**2 + y**2) / 0.3) * np.sin(2 * np.pi * (x + y))
        raw_frame += structure.astype(np.float32)
        processed_frame += structure.astype(np.float32)
        
        frames = {
            "raw": raw_frame,
            "processed": processed_frame
        }
        
        # Generate simulated metadata
        metadata = {
            "exposure_time_ms": np.random.uniform(10, 100),
            "gain": np.random.uniform(1.0, 4.0),
            "temperature_C": np.random.uniform(20, 25),
            "humidity_percent": np.random.uniform(40, 60),
            "pressure_mbar": np.random.uniform(1010, 1020),
            "timestamp": time.time(),
            "frame_number": self.frame_counter
        }
        
        self.frame_counter += 1
        
        return frames, metadata
    
    def _regulate_data(self, frames: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], int, str]:
        """
        Regulate dummy data format for viewer compatibility.
        
        Args:
            frames: Raw frames from dummy source
            metadata: Raw metadata from dummy source
            
        Returns:
            Tuple of (processed_frames, processed_metadata, shot_id, shot_name)
        """
        processed_frames = {}
        processed_metadata = metadata.copy()
        
        # Process each frame
        for frame_name, frame_data in frames.items():
            if frame_data is None:
                continue
                
            # Ensure frame is a numpy array
            if not isinstance(frame_data, np.ndarray):
                frame_data = np.array(frame_data)
            
            # Validate frame dimensions
            if frame_data.ndim != 2:
                logger.warning(f"Frame {frame_name} has {frame_data.ndim} dimensions, expected 2D")
                continue
            
            # Convert to float32 for viewer compatibility
            if frame_data.dtype != np.float32:
                frame_data = frame_data.astype(np.float32, copy=False)
            
            # Store processed frame
            processed_frames[frame_name] = frame_data
        
        # Generate shot information
        self.shot_counter += 1
        shot_id = self.shot_counter
        shot_name = f"Dummy_{shot_id}"
        
        return processed_frames, processed_metadata, shot_id, shot_name


# Convenience function for creating grabbers
def create_grabber(grabber_type: str, store, **kwargs) -> DataGrabber:
    """
    Factory function to create data grabbers.
    
    Args:
        grabber_type: Type of grabber ('zmq', 'pulsar', etc.)
        store: Reference to the viewer's data store
        **kwargs: Additional arguments for the specific grabber type
    
    Returns:
        Configured DataGrabber instance
    """
    if grabber_type.lower() == 'zmq':
        return ZMQDataGrabber(store, **kwargs)
    elif grabber_type.lower() == 'dummy':
        return DummyGrabber(store, **kwargs)
    # TODO: Add other grabber types here
    # elif grabber_type.lower() == 'pulsar':
    #     return PulsarDataGrabber(store, **kwargs)
    else:
        raise ValueError(f"Unknown grabber type: {grabber_type}")
