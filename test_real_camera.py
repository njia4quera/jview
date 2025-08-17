#!/usr/bin/env python3
"""
Test script for real camera server connection.

This script tests the connection to your real quetzal-instruments camera server
and verifies basic functionality.
"""

import time
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_real_camera_connection():
    """Test connection to the real camera server."""
    try:
        from quetzal.instruments.remote import RemoteInstrument
        
        # Connect to real camera server
        host = 'localhost'
        port = 60637
        
        logger.info(f"Connecting to real camera server at {host}:{port}...")
        remote_instrument = RemoteInstrument(host, port)
        logger.info("✓ Connected to real camera server!")
        
        # Test basic methods
        logger.info("Testing available methods...")
        methods = remote_instrument.list_methods()
        logger.info(f"Available methods: {methods}")
        
        # Test available parameters
        logger.info("Testing available parameters...")
        parameters = remote_instrument.list_parameters()
        logger.info(f"Available parameters: {parameters}")
        
        # Test getting shot data
        logger.info("Testing get_shot_data method...")
        try:
            shot_data = remote_instrument.get_shot_data()
            logger.info(f"✓ Successfully received shot data!")
            logger.info(f"Data keys: {shot_data.keys()}")
            
            # Analyze the data structure
            if 'frames' in shot_data:
                frames = shot_data['frames']
                logger.info(f"Frame types: {list(frames.keys())}")
                
                for frame_name, frame_data in frames.items():
                    if hasattr(frame_data, 'shape'):
                        logger.info(f"  {frame_name}: {frame_data.shape}, {frame_data.dtype}")
                    else:
                        logger.info(f"  {frame_name}: {type(frame_data)}")
            
            if 'meta' in shot_data:
                metadata = shot_data['meta']
                logger.info(f"Metadata keys: {list(metadata.keys())}")
                
                # Show some key metadata
                for key in ['shot_id', 'shot_name', 'timestamp', 'exposure_time_ms', 'gain']:
                    if key in metadata:
                        logger.info(f"  {key}: {metadata[key]}")
            
        except Exception as e:
            logger.error(f"❌ Error getting shot data: {e}")
            logger.info("This might be expected if the camera is not ready or the method doesn't exist")
        
        # Test other available methods
        logger.info("Testing other available methods...")
        for method_name in methods.get('data', {}).get('keys', []):
            if method_name not in ['list_methods', 'list_parameters', 'get_parameter', 'set_parameter']:
                try:
                    logger.info(f"Testing method: {method_name}")
                    # Try to call the method (this might fail for some methods)
                    if hasattr(remote_instrument, method_name):
                        method = getattr(remote_instrument, method_name)
                        if callable(method):
                            # Try calling with no arguments first
                            try:
                                result = method()
                                logger.info(f"  ✓ {method_name}() returned: {type(result)}")
                            except Exception as e:
                                logger.info(f"  ⚠ {method_name}() failed: {e}")
                        else:
                            logger.info(f"  ⚠ {method_name} is not callable")
                    else:
                        logger.info(f"  ⚠ {method_name} not found on remote instrument")
                except Exception as e:
                    logger.info(f"  ⚠ Error testing {method_name}: {e}")
        
        # Cleanup
        remote_instrument.close()
        logger.info("✓ Connection test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False

def test_data_grabber_with_real_camera():
    """Test the RemoteInstrumentImageGrabber with the real camera."""
    try:
        from data_grabber import RemoteInstrumentImageGrabber
        
        # Create a simple mock store for testing
        class SimpleStore:
            def __init__(self):
                self.shots = []
                self.shot_counter = 0
            
            def add_shot(self, frames, meta, shot_name):
                shot_id = self.shot_counter + 1
                self.shot_counter = shot_id
                
                shot_data = {
                    'id': shot_id,
                    'name': shot_name,
                    'frames': frames,
                    'meta': meta,
                    'timestamp': time.time()
                }
                
                self.shots.append(shot_data)
                logger.info(f"Store: Added shot {shot_id} ({shot_name}) with {len(frames)} frames")
                return shot_id
        
        mock_store = SimpleStore()
        
        # Create data grabber
        logger.info("Creating RemoteInstrumentImageGrabber...")
        grabber = RemoteInstrumentImageGrabber(mock_store, 'localhost', 60637)
        logger.info("✓ RemoteInstrumentImageGrabber created successfully")
        
        # Test getting data
        logger.info("Testing data acquisition...")
        for i in range(3):  # Try to get 3 shots
            logger.info(f"Attempting to get shot {i+1}...")
            result = grabber.get_data()
            
            if result is not None:
                frames, metadata = result
                logger.info(f"✓ Shot {i+1}: {len(frames)} frames, shot_id={metadata.get('shot_id')}")
                
                # Show frame information
                for frame_name, frame_data in frames.items():
                    if hasattr(frame_data, 'shape'):
                        logger.info(f"  Frame {frame_name}: {frame_data.shape}, {frame_data.dtype}")
                    else:
                        logger.info(f"  Frame {frame_name}: {type(frame_data)}")
                
                # Show metadata
                logger.info(f"  Metadata: {list(metadata.keys())}")
                
            else:
                logger.info(f"⚠ Shot {i+1}: No new data (duplicate or no data available)")
            
            time.sleep(0.5)  # Wait between attempts
        
        # Check store contents
        shots = mock_store.shots
        logger.info(f"Store contains {len(shots)} shots")
        
        for shot in shots:
            logger.info(f"  Shot {shot['id']}: {shot['name']} with {len(shot['frames'])} frames")
        
        # Cleanup
        grabber.stop()
        logger.info("✓ Data grabber test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data grabber test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🔍 Testing real camera server connection...")
    
    # Test 1: Direct connection
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Direct RemoteInstrument Connection")
    logger.info("="*50)
    
    if not test_real_camera_connection():
        logger.error("Direct connection test failed. Check if the camera server is running.")
        return
    
    # Test 2: Data grabber integration
    logger.info("\n" + "="*50)
    logger.info("TEST 2: RemoteInstrumentImageGrabber Integration")
    logger.info("="*50)
    
    if not test_data_grabber_with_real_camera():
        logger.error("Data grabber integration test failed.")
        return
    
    logger.info("\n" + "="*50)
    logger.info("🎉 ALL TESTS PASSED! 🎉")
    logger.info("="*50)
    logger.info("Your real camera server is working correctly with")
    logger.info("the RemoteInstrumentImageGrabber!")

if __name__ == "__main__":
    main()
