#!/usr/bin/env python3
"""
Dummy Camera Server for Testing RemoteInstrumentImageGrabber

This script creates a mock quetzal-instruments server that generates fake camera data
to test the RemoteInstrumentImageGrabber implementation.
"""

import time
import logging
import threading
import numpy as np
from typing import Dict, Any
import zmq
import msgpack

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DummyCameraInstrument:
    """
    Mock camera instrument that generates fake data.
    
    This class simulates a real camera instrument with configurable parameters
    and fake image generation capabilities.
    """
    
    def __init__(self):
        """Initialize the dummy camera instrument."""
        self.exposure_time = 100.0  # ms
        self.gain = 2.0
        self.temperature = 23.5  # Celsius
        self.humidity = 45.0  # percent
        self.pressure = 1013.25  # mbar
        self.frame_counter = 0
        self.last_shot_time = time.time()
        
        # Image generation parameters
        self.image_width = 640
        self.image_height = 480
        self.noise_level = 50
        self.signal_strength = 1000
        
        logger.info(f"Dummy camera initialized: {self.image_width}x{self.image_height}")
    
    def get_shot_data(self) -> Dict[str, Any]:
        """
        Generate fake shot data with simulated images.
        
        Returns:
            Dictionary with 'frames' and 'meta' keys
        """
        self.frame_counter += 1
        current_time = time.time()
        
        # Generate fake images
        frames = self._generate_frames()
        
        # Generate metadata
        metadata = {
            'shot_id': self.frame_counter,
            'shot_name': f'Dummy_Shot_{self.frame_counter:04d}',
            'timestamp': current_time,
            'exposure_time_ms': self.exposure_time,
            'gain': self.gain,
            'temperature_C': self.temperature + np.random.normal(0, 0.1),
            'humidity_percent': self.humidity + np.random.normal(0, 1.0),
            'pressure_mbar': self.pressure + np.random.normal(0, 0.5),
            'frame_number': self.frame_counter,
            'camera_model': 'Dummy_Camera_v1.0',
            'sensor_type': 'Fake_CCD_640x480',
            'bit_depth': 16,
            'pixel_size_um': 6.45,
            'readout_rate_mhz': 10.0,
            'dark_current_e_per_s': 0.1,
            'readout_noise_e': 5.0
        }
        
        # Add some random variation to make shots different
        self._add_random_variation(metadata)
        
        self.last_shot_time = current_time
        
        return {
            'frames': frames,
            'meta': metadata
        }
    
    def _generate_frames(self) -> Dict[str, np.ndarray]:
        """Generate fake camera frames."""
        frames = {}
        
        # Generate raw frame with simulated physics
        raw_frame = self._generate_raw_frame()
        frames['raw'] = raw_frame
        
        # Generate processed frame (simulated analysis)
        processed_frame = self._generate_processed_frame(raw_frame)
        frames['processed'] = processed_frame
        
        # Generate dark frame occasionally
        if self.frame_counter % 10 == 0:
            dark_frame = self._generate_dark_frame()
            frames['dark'] = dark_frame
        
        return frames
    
    def _generate_raw_frame(self) -> np.ndarray:
        """Generate a raw camera frame with simulated physics."""
        # Base noise floor
        frame = np.random.normal(100, self.noise_level, (self.image_height, self.image_width))
        
        # Add simulated atomic cloud structure
        x, y = np.meshgrid(
            np.linspace(-1, 1, self.image_width),
            np.linspace(-1, 1, self.image_height)
        )
        
        # Simulate cold atom cloud (Gaussian with some structure)
        cloud_center_x = np.random.normal(0, 0.3)
        cloud_center_y = np.random.normal(0, 0.3)
        
        # Main cloud
        cloud = self.signal_strength * np.exp(
            -((x - cloud_center_x)**2 + (y - cloud_center_y)**2) / 0.2
        )
        
        # Add some interference fringes
        fringes = 200 * np.sin(2 * np.pi * (x + y) * 2) * np.exp(
            -((x - cloud_center_x)**2 + (y - cloud_center_y)**2) / 0.4
        )
        
        # Combine signals
        frame += cloud + fringes
        
        # Add some random hot pixels
        num_hot_pixels = np.random.poisson(3)
        for _ in range(num_hot_pixels):
            px, py = np.random.randint(0, self.image_width), np.random.randint(0, self.image_height)
            frame[py, px] += np.random.exponential(500)
        
        # Ensure positive values and convert to uint16
        frame = np.clip(frame, 0, 65535).astype(np.uint16)
        
        return frame
    
    def _generate_processed_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Generate a processed frame from raw data."""
        # Simulate basic image processing
        processed = raw_frame.astype(np.float32)
        
        # Background subtraction (simplified)
        background = np.percentile(processed, 10)
        processed -= background
        
        # Noise reduction (simple smoothing)
        from scipy.ndimage import gaussian_filter
        try:
            processed = gaussian_filter(processed, sigma=0.5)
        except ImportError:
            # Fallback if scipy not available
            pass
        
        # Normalize and convert back to uint16
        processed = np.clip(processed, 0, 65535).astype(np.uint16)
        
        return processed
    
    def _generate_dark_frame(self) -> np.ndarray:
        """Generate a dark frame for calibration."""
        # Dark frame has lower noise and no signal
        dark = np.random.normal(50, self.noise_level * 0.5, (self.image_height, self.image_width))
        dark = np.clip(dark, 0, 65535).astype(np.uint16)
        return dark
    
    def _add_random_variation(self, metadata: Dict[str, Any]):
        """Add random variation to metadata to simulate real conditions."""
        # Vary exposure time slightly
        metadata['exposure_time_ms'] += np.random.normal(0, 2.0)
        
        # Vary gain slightly
        metadata['gain'] += np.random.normal(0, 0.1)
        
        # Add some random experimental parameters
        metadata['magnetic_field_gauss'] = np.random.normal(0, 0.1)
        metadata['laser_power_mw'] = np.random.normal(100, 5.0)
        metadata['trap_frequency_hz'] = np.random.normal(1000, 50.0)
        metadata['atom_number'] = int(np.random.normal(10000, 1000))
        metadata['temperature_nk'] = np.random.normal(100, 10.0)


class DummyQuetzalServer:
    """
    Mock quetzal-instruments server that exposes the dummy camera.
    
    This class implements a simple ZMQ-based server that mimics the
    quetzal-instruments server interface.
    """
    
    def __init__(self, bind_address: str = "*", port: int = 8700):
        """
        Initialize the dummy quetzal server.
        
        Args:
            bind_address: Address to bind to (default: all interfaces)
            port: Port to listen on (default: 8700)
        """
        self.bind_address = bind_address
        self.port = port
        self.running = False
        
        # Create ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        
        # Create dummy camera instrument
        self.camera = DummyCameraInstrument()
        
        # Server parameters
        self.parameters = {
            'version': '0.33.7',
            'name': 'DummyCameraInstrument',
            'api_version': '0.33.7',
            'server_language': 'python',
            'python_version': f'{3}.{9}',
            'platform': 'linux'
        }
        
        logger.info(f"Dummy quetzal server initialized on {bind_address}:{port}")
    
    def start(self):
        """Start the server."""
        try:
            # Bind socket
            bind_url = f"tcp://{self.bind_address}:{self.port}"
            self.socket.bind(bind_url)
            self.running = True
            
            logger.info(f"Server started on {bind_url}")
            logger.info("Press Ctrl+C to stop the server")
            
            # Start server loop
            self._run_server()
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()
        logger.info("Server stopped")
    
    def _run_server(self):
        """Main server loop."""
        while self.running:
            try:
                # Wait for request
                request = self.socket.recv(flags=zmq.NOBLOCK)
                response = self._handle_request(request)
                
                # Send response
                self.socket.send(response)
                
            except zmq.Again:
                # No message available, continue
                time.sleep(0.001)
                continue
            except Exception as e:
                logger.error(f"Error handling request: {e}")
                # Send error response
                error_response = msgpack.packb({
                    'error': {'message': str(e)}
                })
                try:
                    self.socket.send(error_response)
                except:
                    pass
    
    def _handle_request(self, request: bytes) -> bytes:
        """Handle incoming requests and generate responses."""
        try:
            # Parse request
            payload = msgpack.unpackb(request)
            call = payload.get('call', '')
            data = payload.get('data', {})
            
            logger.debug(f"Received request: {call}")
            
            # Handle different method calls
            if call == 'list_methods':
                response = self._list_methods()
            elif call == 'list_parameters':
                response = self._list_parameters()
            elif call == 'get_parameter':
                response = self._get_parameter(data)
            elif call == 'set_parameter':
                response = self._set_parameter(data)
            elif call == 'get_shot_data':
                response = self._get_shot_data()
            else:
                response = {
                    'error': {'message': f'Unknown method: {call}'}
                }
            
            return msgpack.packb(response)
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return msgpack.packb({
                'error': {'message': str(e)}
            })
    
    def _list_methods(self) -> Dict[str, Any]:
        """List available methods."""
        return {
            'call': 'list_methods',
            'data': {
                'keys': [
                    'list_methods',
                    'list_parameters',
                    'get_parameter',
                    'set_parameter',
                    'get_shot_data'
                ]
            }
        }
    
    def _list_parameters(self) -> Dict[str, Any]:
        """List available parameters."""
        return {
            'call': 'list_parameters',
            'data': {
                'keys': list(self.parameters.keys())
            }
        }
    
    def _get_parameter(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameter value."""
        key = data.get('key', '')
        if key in self.parameters:
            return {
                'call': 'get_parameter',
                'data': {
                    'key': key,
                    'value': self.parameters[key]
                }
            }
        else:
            return {
                'error': {'message': f'Unknown parameter: {key}'}
            }
    
    def _set_parameter(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Set parameter value."""
        key = data.get('key', '')
        value = data.get('value')
        
        if key in self.parameters:
            self.parameters[key] = value
            return {'call': 'set_parameter'}
        else:
            return {
                'error': {'message': f'Unknown parameter: {key}'}
            }
    
    def _get_shot_data(self) -> Dict[str, Any]:
        """Get shot data from dummy camera."""
        shot_data = self.camera.get_shot_data()
        return {
            'call': 'get_shot_data',
            'data': shot_data
        }


def main():
    """Main function to run the dummy camera server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dummy Camera Server for Testing')
    parser.add_argument('--host', default='*', help='Host to bind to (default: *)')
    parser.add_argument('--port', type=int, default=8700, help='Port to listen on (default: 8700)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start server
    server = DummyQuetzalServer(args.host, args.port)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
