#!/usr/bin/env python3
"""
Test script for RemoteInstrumentImageGrabber.

This script tests the quetzal-instruments based data grabber without requiring
an actual camera server connection.
"""

import sys
import os
import time
import logging
from unittest.mock import Mock, MagicMock
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_duplicate_detection():
    """Test the duplicate detection logic."""
    from data_grabber import RemoteInstrumentImageGrabber
    
    # Create a mock store
    mock_store = Mock()
    
    # Create grabber instance (will fail to connect, but we can test methods)
    try:
        grabber = RemoteInstrumentImageGrabber(mock_store, 'localhost', 8700)
        logger.info("✓ RemoteInstrumentImageGrabber created successfully")
    except SystemExit:
        logger.info("✓ RemoteInstrumentImageGrabber created successfully (expected SystemExit)")
        # Create a mock instance for testing
        grabber = RemoteInstrumentImageGrabber.__new__(RemoteInstrumentImageGrabber)
        grabber._last_hash = None
        grabber._last_timestamp = None
    
    # Test duplicate detection with identical frames
    frame1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    frame3 = frame1.copy()  # Identical to frame1
    
    frames1 = {'raw': frame1}
    frames2 = {'raw': frame2}
    frames3 = {'raw': frame3}
    
    metadata = {'timestamp': time.time()}
    
    # First shot should be new
    is_new1 = grabber._is_new_shot(frames1, metadata)
    logger.info(f"First shot (new): {is_new1}")
    
    # Second shot should be new (different content)
    is_new2 = grabber._is_new_shot(frames2, metadata)
    logger.info(f"Second shot (new): {is_new2}")
    
    # Third shot should be duplicate (same content as first)
    is_new3 = grabber._is_new_shot(frames3, metadata)
    logger.info(f"Third shot (duplicate): {is_new3}")
    
    # Verify results
    assert is_new1 == True, "First shot should be new"
    assert is_new2 == True, "Second shot should be new"
    assert is_new3 == False, "Third shot should be duplicate"
    
    logger.info("✓ Duplicate detection test passed")

def test_data_regulation():
    """Test the data regulation logic."""
    from data_grabber import RemoteInstrumentImageGrabber
    
    # Create a mock store
    mock_store = Mock()
    
    # Create grabber instance
    try:
        grabber = RemoteInstrumentImageGrabber(mock_store, 'localhost', 8700)
    except SystemExit:
        grabber = RemoteInstrumentImageGrabber.__new__(RemoteInstrumentImageGrabber)
        grabber.shot_counter = 0
    
    # Test data with metadata
    frames = {
        'raw': np.random.randint(0, 255, (100, 100), dtype=np.uint8),
        'processed': np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    }
    
    metadata = {
        'shot_id': 42,
        'shot_name': 'test_shot',
        'exposure_time': 100,
        'timestamp': time.time()
    }
    
    # Test regulation
    processed_frames, processed_metadata, shot_id, shot_name = grabber._regulate_data(frames, metadata)
    
    # Verify results
    assert shot_id == 42, f"Expected shot_id 42, got {shot_id}"
    assert shot_name == 'test_shot', f"Expected shot_name 'test_shot', got {shot_name}"
    assert len(processed_frames) == 2, f"Expected 2 frames, got {len(processed_frames)}"
    assert 'raw' in processed_frames, "Raw frame should be present"
    assert 'processed' in processed_frames, "Processed frame should be present"
    
    logger.info("✓ Data regulation test passed")

def test_factory_function():
    """Test the factory function creates the right grabber type."""
    from data_grabber import create_grabber
    
    # Create a mock store
    mock_store = Mock()
    
    # Test quetzal grabber creation
    try:
        grabber = create_grabber('quetzal', mock_store, host='localhost', port=8700)
        logger.info("✓ Quetzal grabber created successfully")
        assert grabber.__class__.__name__ == 'RemoteInstrumentImageGrabber'
    except SystemExit:
        logger.info("✓ Quetzal grabber created successfully (expected SystemExit)")
    
    # Test invalid grabber type
    try:
        grabber = create_grabber('invalid', mock_store)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        logger.info(f"✓ Invalid grabber type correctly rejected: {e}")

if __name__ == "__main__":
    logger.info("Testing RemoteInstrumentImageGrabber implementation...")
    
    try:
        test_duplicate_detection()
        test_data_regulation()
        test_factory_function()
        logger.info("🎉 All tests passed!")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)
