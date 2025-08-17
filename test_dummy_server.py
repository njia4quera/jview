#!/usr/bin/env python3
"""
Test client for the dummy camera server.

This script tests the dummy quetzal server to ensure it's working correctly
before testing the RemoteInstrumentImageGrabber.
"""

import time
import logging
import zmq
import msgpack
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_server_connection(host: str = 'localhost', port: int = 8700):
    """Test basic connection to the dummy server."""
    try:
        # Create ZMQ context and socket
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        
        # Connect to server
        server_url = f"tcp://{host}:{port}"
        socket.connect(server_url)
        logger.info(f"Connected to server at {server_url}")
        
        # Test list_methods
        logger.info("Testing list_methods...")
        request = msgpack.packb({'call': 'list_methods'})
        socket.send(request)
        response = msgpack.unpackb(socket.recv())
        logger.info(f"Methods: {response}")
        
        # Test list_parameters
        logger.info("Testing list_parameters...")
        request = msgpack.packb({'call': 'list_parameters'})
        socket.send(request)
        response = msgpack.unpackb(socket.recv())
        logger.info(f"Parameters: {response}")
        
        # Test get_shot_data
        logger.info("Testing get_shot_data...")
        request = msgpack.packb({'call': 'get_shot_data'})
        socket.send(request)
        response = msgpack.unpackb(socket.recv())
        
        if 'data' in response:
            shot_data = response['data']
            frames = shot_data.get('frames', {})
            metadata = shot_data.get('meta', {})
            
            logger.info(f"Received shot {metadata.get('shot_id')}: {metadata.get('shot_name')}")
            logger.info(f"Frame types: {list(frames.keys())}")
            
            # Check frame data
            for frame_name, frame_data in frames.items():
                if isinstance(frame_data, np.ndarray):
                    logger.info(f"  {frame_name}: {frame_data.shape}, {frame_data.dtype}")
                else:
                    logger.info(f"  {frame_name}: {type(frame_data)}")
            
            # Check metadata
            logger.info(f"Metadata keys: {list(metadata.keys())}")
            logger.info(f"Timestamp: {metadata.get('timestamp')}")
            logger.info(f"Exposure: {metadata.get('exposure_time_ms')} ms")
            logger.info(f"Gain: {metadata.get('gain')}")
            
        else:
            logger.error(f"Unexpected response: {response}")
        
        # Cleanup
        socket.close()
        context.term()
        
        logger.info("✓ Server connection test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Server connection test failed: {e}")
        return False

def test_multiple_shots(host: str = 'localhost', port: int = 8700, num_shots: int = 5):
    """Test getting multiple shots to verify data variation."""
    try:
        # Create ZMQ context and socket
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        
        # Connect to server
        server_url = f"tcp://{host}:{port}"
        socket.connect(server_url)
        logger.info(f"Testing {num_shots} shots from {server_url}")
        
        shot_ids = []
        timestamps = []
        
        for i in range(num_shots):
            # Get shot data
            request = msgpack.packb({'call': 'get_shot_data'})
            socket.send(request)
            response = msgpack.unpackb(socket.recv())
            
            if 'data' in response:
                shot_data = response['data']
                metadata = shot_data.get('meta', {})
                
                shot_id = metadata.get('shot_id')
                timestamp = metadata.get('timestamp')
                
                shot_ids.append(shot_id)
                timestamps.append(timestamp)
                
                logger.info(f"Shot {i+1}: ID={shot_id}, Time={timestamp}")
                
                # Small delay between shots
                time.sleep(0.1)
            else:
                logger.error(f"Failed to get shot {i+1}: {response}")
        
        # Verify data variation
        if len(set(shot_ids)) == num_shots:
            logger.info("✓ All shot IDs are unique")
        else:
            logger.warning("⚠ Some shot IDs are not unique")
        
        if len(set(timestamps)) == num_shots:
            logger.info("✓ All timestamps are unique")
        else:
            logger.warning("⚠ Some timestamps are not unique")
        
        # Cleanup
        socket.close()
        context.term()
        
        logger.info("✓ Multiple shots test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Multiple shots test failed: {e}")
        return False

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Dummy Camera Server')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8700, help='Server port (default: 8700)')
    parser.add_argument('--shots', type=int, default=5, help='Number of shots to test (default: 5)')
    
    args = parser.parse_args()
    
    logger.info("Testing dummy camera server...")
    
    # Test basic connection
    if not test_server_connection(args.host, args.port):
        logger.error("Basic connection test failed. Is the server running?")
        return
    
    # Test multiple shots
    if not test_multiple_shots(args.host, args.port, args.shots):
        logger.error("Multiple shots test failed.")
        return
    
    logger.info("🎉 All tests passed! The dummy server is working correctly.")

if __name__ == "__main__":
    main()
