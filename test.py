#!/usr/bin/env python3
"""
Test file for jviewer app
Creates fake fluorescence data shots with metadata for testing
"""

import numpy as np
import time
import threading
import subprocess
import sys
import os

# Import the app instance to access app.data_pool
from app import app, add_new_shot, get_pool_status, get_meta_dataframe

def create_fake_fluorescence_data(shot_number, image_size=(256, 256)):
    """
    Create fake fluorescence data for testing
    
    Args:
        shot_number (int): Shot number for generating unique data
        image_size (tuple): Size of the images (height, width)
    
    Returns:
        tuple: (frames_data, meta_data)
    """
    height, width = image_size
    
    # Create base patterns that vary with shot number
    x = np.linspace(0, 2*np.pi, width)
    y = np.linspace(0, 2*np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    # Add some shot-to-shot variation
    phase_shift = shot_number * 0.5
    intensity_factor = 1.0 + 0.2 * np.sin(shot_number * 0.3)
    
    # Create foreground (signal + noise)
    signal = np.sin(X + phase_shift) * np.cos(Y + phase_shift) * intensity_factor
    noise = np.random.normal(0, 0.1, (height, width))
    foreground = np.clip(signal + noise, 0, None)  # Ensure non-negative
    
    # Create background (baseline + drift)
    background_baseline = 0.3 + 0.1 * np.sin(shot_number * 0.2)
    background_drift = np.random.normal(0, 0.05, (height, width))
    background = background_baseline + background_drift
    
    # Create subtracted (foreground - background)
    subtracted = foreground - background
    
    # Add some random hot spots
    num_hotspots = np.random.poisson(3)  # Random number of hot spots
    for _ in range(num_hotspots):
        hotspot_y = np.random.randint(0, height)
        hotspot_x = np.random.randint(0, width)
        radius = np.random.randint(5, 15)
        intensity = np.random.uniform(0.5, 2.0)
        
        # Create Gaussian hotspot
        for i in range(max(0, hotspot_y-radius), min(height, hotspot_y+radius)):
            for j in range(max(0, hotspot_x-radius), min(width, hotspot_x+radius)):
                dist = np.sqrt((i-hotspot_y)**2 + (j-hotspot_x)**2)
                if dist < radius:
                    gaussian = intensity * np.exp(-(dist**2) / (2 * (radius/3)**2))
                    foreground[i, j] += gaussian
                    subtracted[i, j] += gaussian
    
    # Create multiple frames for each type (simulating time series)
    num_frames = np.random.randint(3, 8)  # Random number of frames
    
    frames_data = {
        "fluorescence_subtracted": [subtracted + np.random.normal(0, 0.02, (height, width)) for _ in range(num_frames)],
        "fluorescence_foreground": [foreground + np.random.normal(0, 0.02, (height, width)) for _ in range(num_frames)],
        "fluorescence_background": [background + np.random.normal(0, 0.02, (height, width)) for _ in range(num_frames)]
    }
    
    # Create comprehensive metadata
    meta_data = {
        "shot_number": shot_number,
        "timestamp": time.time() + shot_number,  # Simulate time progression
        "laser_power_mw": np.random.uniform(50, 200),
        "exposure_time_ms": np.random.uniform(10, 100),
        "camera_gain": np.random.uniform(1.0, 5.0),
        "temperature_celsius": np.random.uniform(20, 25),
        "humidity_percent": np.random.uniform(30, 60),
        "pressure_mbar": np.random.uniform(1000, 1020),
        "fluorescence_intensity_avg": np.mean(foreground),
        "fluorescence_intensity_max": np.max(foreground),
        "background_level": np.mean(background),
        "signal_to_noise_ratio": np.mean(foreground) / (np.std(noise) + 1e-6),
        "hotspot_count": num_hotspots,
        "image_width": width,
        "image_height": height,
        "num_frames": num_frames,
        "shot_quality_score": np.random.uniform(0.7, 1.0),
        "laser_wavelength_nm": np.random.choice([488, 532, 633, 780]),
        "filter_bandwidth_nm": np.random.uniform(10, 50),
        "sample_concentration_um": np.random.uniform(0.1, 10.0),
        "sample_volume_ul": np.random.uniform(10, 100),
        "experiment_duration_min": np.random.uniform(5, 60),
        "operator_id": f"user_{np.random.randint(1, 5)}",
        "experiment_type": np.random.choice(["fluorescence", "absorption", "scattering"]),
        "sample_preparation_time_min": np.random.uniform(10, 120),
        "calibration_date": f"2024-{np.random.randint(1, 12):02d}-{np.random.randint(1, 28):02d}",
        "instrument_id": f"inst_{np.random.randint(1, 10):03d}",
        "data_quality_flag": np.random.choice([0, 1, 2]),  # 0=good, 1=warning, 2=error
        "processing_version": f"v{np.random.randint(1, 5)}.{np.random.randint(0, 9)}",
        "file_size_mb": np.random.uniform(0.5, 5.0),
        "acquisition_rate_fps": np.random.uniform(1, 30),
        "trigger_delay_ms": np.random.uniform(0, 100),
        "integration_time_ms": np.random.uniform(1, 50),
        "dark_current_adu": np.random.uniform(0, 100),
        "readout_noise_electrons": np.random.uniform(1, 10),
        "quantum_efficiency": np.random.uniform(0.3, 0.9),
        "pixel_size_um": np.random.uniform(3.0, 20.0),
        "magnification": np.random.uniform(10, 100),
        "numerical_aperture": np.random.uniform(0.1, 1.4),
        "working_distance_mm": np.random.uniform(0.1, 10.0),
        "focus_position_um": np.random.uniform(-100, 100),
        "stage_x_mm": np.random.uniform(-10, 10),
        "stage_y_mm": np.random.uniform(-10, 10),
        "stage_z_mm": np.random.uniform(-5, 5),
        "objective_immersion": np.random.choice(["air", "water", "oil"]),
        "sample_mounting": np.random.choice(["slide", "coverslip", "well_plate", "flow_cell"]),
        "sample_temperature_celsius": np.random.uniform(15, 37),
        "sample_ph": np.random.uniform(6.0, 8.0),
        "buffer_concentration_mm": np.random.uniform(10, 150),
        "excitation_power_density_w_cm2": np.random.uniform(0.1, 10.0),
        "emission_collection_efficiency": np.random.uniform(0.1, 0.8),
        "detector_dark_current_adu": np.random.uniform(0, 50),
        "detector_gain_electrons_per_adu": np.random.uniform(0.5, 10.0),
        "detector_readout_speed_mhz": np.random.uniform(1, 100),
        "detector_cooling_temperature_celsius": np.random.uniform(-80, -20),
        "detector_quantum_efficiency_at_wavelength": np.random.uniform(0.1, 0.9),
        "optical_path_length_cm": np.random.uniform(1, 50),
        "beam_diameter_um": np.random.uniform(1, 100),
        "beam_profile": np.random.choice(["gaussian", "top_hat", "donut"]),
        "polarization_angle_degrees": np.random.uniform(0, 180),
        "modulation_frequency_hz": np.random.uniform(0, 1000),
        "lock_in_time_constant_ms": np.random.uniform(1, 1000),
        "reference_signal_adu": np.random.uniform(100, 1000),
        "background_subtraction_method": np.random.choice(["frame_difference", "rolling_average", "manual"]),
        "smoothing_applied": np.random.choice([True, False]),
        "smoothing_kernel_size": np.random.randint(3, 11) if np.random.random() > 0.5 else 0,
        "normalization_applied": np.random.choice([True, False]),
        "normalization_method": np.random.choice(["max", "sum", "mean", "none"]),
        "outlier_removal": np.random.choice([True, False]),
        "outlier_threshold_sigma": np.random.uniform(2, 5) if np.random.random() > 0.5 else 0,
        "data_compression": np.random.choice([True, False]),
        "compression_ratio": np.random.uniform(1, 10) if np.random.random() > 0.5 else 0,
        "metadata_complete": True,
        "calibration_valid": np.random.choice([True, False]),
        "quality_control_passed": np.random.choice([True, False]),
        "archived": np.random.choice([True, False]),
        "backup_location": np.random.choice(["local", "cloud", "tape", "none"]),
        "data_retention_days": np.random.randint(30, 3650),
        "access_level": np.random.choice(["public", "internal", "restricted", "confidential"]),
        "last_modified": time.time(),
        "checksum": f"sha256_{np.random.randint(1000000, 9999999)}",
        "version": "1.0.0"
    }
    
    return frames_data, meta_data

def real_time_data_generator(shot_interval=2.0, max_shots=None):
    """
    Continuously generate new shots to simulate real-time incoming data
    
    Args:
        shot_interval (float): Time between shots in seconds
        max_shots (int): Maximum number of shots to generate (None for infinite)
    """
    shot_number = 1
    start_time = time.time()
    
    print(f"Starting real-time data generation...")
    print(f"New shot every {shot_interval} seconds")
    print(f"Press Ctrl+C to stop")
    
    try:
        while max_shots is None or shot_number <= max_shots:
            print(f"Generating shot {shot_number} at {time.strftime('%H:%M:%S')}...")
            
            # Create fake data
            frames_data, meta_data = create_fake_fluorescence_data(shot_number)
            
            # Add to the app's data pool
            add_new_shot(frames_data, meta_data)
            
            # Print some status info
            status = get_pool_status()
            print(f"  Pool: {status['frame_count']} frames, {status['meta_count']} meta entries")
            print(f"  Data version: {status['data_version']}")
            
            shot_number += 1
            
            # Wait for next shot
            time.sleep(shot_interval)
            
    except KeyboardInterrupt:
        print(f"\nStopped after {shot_number-1} shots")
        print(f"Total runtime: {time.time() - start_time:.1f} seconds")

def test_app_with_fake_data():
    """Test the app by adding multiple fake shots"""
    print("Creating fake fluorescence data shots for testing...")
    
    # Create multiple shots
    num_shots = 10
    for i in range(num_shots):
        print(f"Creating shot {i+1}/{num_shots}...")
        
        # Create fake data
        frames_data, meta_data = create_fake_fluorescence_data(i+1)
        
        # Add to the app's data pool
        add_new_shot(frames_data, meta_data)
        
        # Small delay to simulate real-time acquisition
        time.sleep(0.1)
    
    # Print status
    status = get_pool_status()
    print(f"\nData pool status:")
    print(f"  Frame count: {status['frame_count']}")
    print(f"  Meta count: {status['meta_count']}")
    print(f"  Has data: {status['has_data']}")
    print(f"  Data version: {status['data_version']}")
    
    # Get metadata DataFrame
    meta_df = get_meta_dataframe()
    print(f"\nMetadata columns: {list(meta_df.columns)}")
    print(f"Metadata shape: {meta_df.shape}")
    
    # Show some sample data
    if not meta_df.empty:
        print(f"\nSample metadata from first shot:")
        first_shot = meta_df.iloc[0]
        for key in ['shot_number', 'laser_power_mw', 'exposure_time_ms', 'fluorescence_intensity_avg']:
            if key in first_shot:
                print(f"  {key}: {first_shot[key]}")
    
    print(f"\nTest data created successfully!")
    print(f"Run 'python app.py' to view the data in the web interface.")

def start_realtime_mode():
    """Start the app and begin real-time data generation"""
    print("Starting jviewer in real-time mode...")
    print("This will:")
    print("1. Start the web app in a separate process")
    print("2. Begin generating new shots every 2 seconds")
    print("3. Allow you to view data in real-time in your browser")
    print()
    
    # Start the app in a separate process
    try:
        print("Starting web app...")
        app_process = subprocess.Popen([sys.executable, "app.py"], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
        
        # Wait a moment for the app to start
        print("Waiting for app to start...")
        time.sleep(3)
        
        if app_process.poll() is None:
            print("✓ Web app started successfully!")
            print("Open your browser to: http://127.0.0.1:8050")
            print()
            print("Starting real-time data generation...")
            print("New shots will appear automatically in the web interface")
            print("Press Ctrl+C to stop both the app and data generation")
            print()
            
            # Start generating data
            real_time_data_generator(shot_interval=2.0)
            
        else:
            print("✗ Failed to start web app")
            stdout, stderr = app_process.communicate()
            print(f"Error: {stderr.decode()}")
            
    except KeyboardInterrupt:
        print("\nStopping...")
        if 'app_process' in locals():
            app_process.terminate()
            print("Web app stopped")
    except Exception as e:
        print(f"Error: {e}")
        if 'app_process' in locals():
            app_process.terminate()

def verify_app_connection():
    """Verify that we can connect to the app's data pool"""
    try:
        # Check if we can access the app's data pool
        status = get_pool_status()
        print(f"✓ Successfully connected to app data pool")
        print(f"  Current status: {status}")
        return True
    except Exception as e:
        print(f"✗ Failed to connect to app data pool: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test jviewer app with fake data")
    parser.add_argument("--mode", choices=["test", "realtime"], default="test",
                       help="Mode to run: 'test' for one-time data creation, 'realtime' for continuous data generation")
    parser.add_argument("--interval", type=float, default=2.0,
                       help="Interval between shots in seconds (for realtime mode)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify connection to app data pool before proceeding")
    
    args = parser.parse_args()
    
    # Verify app connection if requested
    if args.verify:
        if not verify_app_connection():
            print("Cannot proceed without app connection. Make sure app.py is running.")
            sys.exit(1)
    
    if args.mode == "realtime":
        start_realtime_mode()
    else:
        test_app_with_fake_data()
