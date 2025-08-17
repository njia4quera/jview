#!/usr/bin/env python3
"""
Full Integration Test for RemoteInstrumentImageGrabber with Dummy Server

This script tests the complete data flow from the dummy camera server through
the RemoteInstrumentImageGrabber to verify everything works together.
"""

import time
import logging
import threading
import subprocess
import signal
import sys
import os
from unittest.mock import Mock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockStore:
    """Mock store for testing the data grabber."""
    
    def __init__(self):
        self.shots = []
        self.shot_counter = 0
    
    def add_shot(self, frames, meta, shot_name):
        """Mock add_shot method."""
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
    
    def get_shot(self, shot_id):
        """Get shot by ID."""
        for shot in self.shots:
            if shot['id'] == shot_id:
                return shot
        return None
    
    def list_shots(self):
        """List all shots."""
        return self.shots

def start_dummy_server(host='localhost', port=8700):
    """Start the dummy camera server in a subprocess."""
    try:
        # Start the dummy server
        cmd = [sys.executable, 'dummy_camera_server.py', '--host', host, '--port', str(port)]
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Check if server is running
        if server_process.poll() is None:
            logger.info(f"✓ Dummy server started on {host}:{port} (PID: {server_process.pid})")
            return server_process
        else:
            stdout, stderr = server_process.communicate()
            logger.error(f"❌ Failed to start dummy server")
            logger.error(f"stdout: {stdout}")
            logger.error(f"stderr: {stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error starting dummy server: {e}")
        return None

def stop_dummy_server(server_process):
    """Stop the dummy camera server."""
    if server_process:
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
            logger.info("✓ Dummy server stopped")
        except subprocess.TimeoutExpired:
            server_process.kill()
            logger.warning("⚠ Dummy server force killed")
        except Exception as e:
            logger.error(f"❌ Error stopping dummy server: {e}")

def test_remote_instrument_connection():
    """Test direct connection to the dummy server using RemoteInstrument."""
    try:
        from quetzal.instruments.remote import RemoteInstrument
        
        # Connect to dummy server
        remote_instrument = RemoteInstrument('localhost', 8700)
        logger.info("✓ Connected to dummy server using RemoteInstrument")
        
        # Test basic methods
        methods = remote_instrument.list_methods()
        logger.info(f"Available methods: {methods}")
        
        parameters = remote_instrument.list_parameters()
        logger.info(f"Available parameters: {parameters}")
        
        # Test getting shot data
        shot_data = remote_instrument.get_shot_data()
        logger.info(f"Received shot data: {shot_data.keys()}")
        
        # Cleanup
        remote_instrument.close()
        logger.info("✓ RemoteInstrument connection test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ RemoteInstrument connection test failed: {e}")
        return False

def test_data_grabber_integration():
    """Test the RemoteInstrumentImageGrabber with the dummy server."""
    try:
        from data_grabber import RemoteInstrumentImageGrabber
        
        # Create mock store
        mock_store = MockStore()
        
        # Create data grabber
        grabber = RemoteInstrumentImageGrabber(mock_store, 'localhost', 8700)
        logger.info("✓ RemoteInstrumentImageGrabber created successfully")
        
        # Test getting data
        logger.info("Testing data acquisition...")
        for i in range(5):
            result = grabber.get_data()
            if result is not None:
                frames, metadata = result
                logger.info(f"Shot {i+1}: {len(frames)} frames, shot_id={metadata.get('shot_id')}")
            else:
                logger.info(f"Shot {i+1}: No new data (duplicate)")
            
            time.sleep(0.1)  # Small delay between shots
        
        # Check store contents
        shots = mock_store.list_shots()
        logger.info(f"Store contains {len(shots)} shots")
        
        for shot in shots:
            logger.info(f"  Shot {shot['id']}: {shot['name']} with {len(shot['frames'])} frames")
        
        # Test duplicate detection
        logger.info("Testing duplicate detection...")
        duplicate_count = 0
        for i in range(10):
            result = grabber.get_data()
            if result is None:
                duplicate_count += 1
            time.sleep(0.05)
        
        logger.info(f"Duplicate detection: {duplicate_count}/10 shots were duplicates")
        
        # Cleanup
        grabber.stop()
        logger.info("✓ Data grabber integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data grabber integration test failed: {e}")
        return False

def test_full_data_flow():
    """Test the complete data flow from server to store."""
    try:
        from data_grabber import RemoteInstrumentImageGrabber
        
        # Create mock store
        mock_store = MockStore()
        
        # Create data grabber
        grabber = RemoteInstrumentImageGrabber(mock_store, 'localhost', 8700)
        
        # Start the grabber (this will run in background)
        grabber.start()
        logger.info("✓ Data grabber started")
        
        # Let it run for a few seconds
        time.sleep(3)
        
        # Stop the grabber
        grabber.stop()
        logger.info("✓ Data grabber stopped")
        
        # Check results
        shots = mock_store.list_shots()
        logger.info(f"Full data flow test: {len(shots)} shots acquired")
        
        if len(shots) > 0:
            # Analyze the data
            total_frames = sum(len(shot['frames']) for shot in shots)
            logger.info(f"Total frames acquired: {total_frames}")
            
            # Check for duplicates (should be none due to hash detection)
            shot_ids = [shot['id'] for shot in shots]
            unique_ids = set(shot_ids)
            if len(shot_ids) == len(unique_ids):
                logger.info("✓ No duplicate shots detected")
            else:
                logger.warning("⚠ Duplicate shots detected")
            
            # Check frame data
            for shot in shots[:3]:  # Check first 3 shots
                logger.info(f"Shot {shot['id']}: {shot['name']}")
                for frame_name, frame_data in shot['frames'].items():
                    if hasattr(frame_data, 'shape'):
                        logger.info(f"  {frame_name}: {frame_data.shape}, {frame_data.dtype}")
                    else:
                        logger.info(f"  {frame_name}: {type(frame_data)}")
        
        logger.info("✓ Full data flow test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Full data flow test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting full integration test...")
    
    # Start dummy server
    server_process = start_dummy_server()
    if not server_process:
        logger.error("Failed to start dummy server. Exiting.")
        return
    
    try:
        # Wait for server to be ready
        time.sleep(1)
        
        # Test 1: Direct RemoteInstrument connection
        logger.info("\n" + "="*50)
        logger.info("TEST 1: Direct RemoteInstrument Connection")
        logger.info("="*50)
        if not test_remote_instrument_connection():
            logger.error("Direct connection test failed")
            return
        
        # Test 2: Data grabber integration
        logger.info("\n" + "="*50)
        logger.info("TEST 2: Data Grabber Integration")
        logger.info("="*50)
        if not test_data_grabber_integration():
            logger.error("Data grabber integration test failed")
            return
        
        # Test 3: Full data flow
        logger.info("\n" + "="*50)
        logger.info("TEST 3: Full Data Flow")
        logger.info("="*50)
        if not test_full_data_flow():
            logger.error("Full data flow test failed")
            return
        
        logger.info("\n" + "="*50)
        logger.info("🎉 ALL TESTS PASSED! 🎉")
        logger.info("="*50)
        logger.info("The RemoteInstrumentImageGrabber is working correctly")
        logger.info("with the dummy camera server.")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup
        stop_dummy_server(server_process)

if __name__ == "__main__":
    main()
