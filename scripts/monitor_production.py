#!/usr/bin/env python
"""Production monitoring script for the video generation system.

This script demonstrates how to integrate the monitoring system with the
video generation pipeline in a production environment.
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pib_videogen import VideoGenerator, VideoGenConfig
from src.monitoring import MonitoringService, AlertLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', f'monitor_production_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)

logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)


def setup_monitoring(base_dir: str = 'data/monitoring') -> MonitoringService:
    """Set up the monitoring service.
    
    Args:
        base_dir: Base directory for monitoring data.
        
    Returns:
        Configured MonitoringService instance.
    """
    # Create monitoring service
    monitoring_service = MonitoringService(
        base_dir=base_dir,
        collection_interval=30,  # Collect metrics every 30 seconds
        snapshot_interval=300,   # Save performance snapshot every 5 minutes
        auto_start=True          # Start monitoring immediately
    )
    
    # Register alert handler
    monitoring_service.register_alert_handler(alert_handler)
    
    # Register metrics update handler
    monitoring_service.register_metrics_update_handler(metrics_update_handler)
    
    logger.info(f"Monitoring service set up with base directory at {base_dir}")
    return monitoring_service


def alert_handler(alert: Dict[str, Any]) -> None:
    """Handle alerts from the monitoring system.
    
    Args:
        alert: Alert dictionary.
    """
    level = alert['level']
    name = alert['name']
    description = alert['description']
    value = alert['actual_value']
    threshold = alert['threshold']
    
    logger.warning(f"ALERT [{level.upper()}]: {name} - {description} (value: {value}, threshold: {threshold})")
    
    # For critical alerts, you might want to send notifications or take action
    if level == AlertLevel.CRITICAL.value:
        # Example: Send notification to admin
        logger.critical(f"CRITICAL ALERT: {description} - Notifying administrators")
        # notify_admin(alert)  # Implement this function if needed


def metrics_update_handler(metrics: Dict[str, Any]) -> None:
    """Handle metrics updates from the monitoring system.
    
    Args:
        metrics: Combined metrics dictionary.
    """
    # This function is called whenever metrics are updated
    # You can use it to update dashboards, store metrics in a database, etc.
    pass


def generate_video_with_monitoring(
    generator: VideoGenerator,
    monitoring_service: MonitoringService,
    prompt: str,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate a video with monitoring.
    
    Args:
        generator: VideoGenerator instance.
        monitoring_service: MonitoringService instance.
        prompt: Text prompt for video generation.
        output_path: Path to save the generated video.
        metadata: Additional metadata about the generation.
        
    Returns:
        Dictionary with generation results.
    """
    # Create metadata if not provided
    if metadata is None:
        metadata = {}
    
    # Add prompt to metadata
    metadata['prompt'] = prompt
    metadata['output_path'] = output_path
    
    # Generate a unique ID for this generation
    generation_id = f"gen_{int(time.time())}_{os.getpid()}"
    
    try:
        # Start tracking generation
        monitoring_service.track_generation_start(generation_id, metadata)
        
        # Generate the video
        start_time = time.time()
        result = generator.generate_video(prompt, output_path)
        generation_time = time.time() - start_time
        
        # Extract quality metrics if available
        quality_metrics = {}
        if 'quality_metrics' in result:
            quality_metrics = result['quality_metrics']
        elif 'metrics' in result:
            quality_metrics = result['metrics']
        
        # If no quality metrics are available, create some basic ones
        if not quality_metrics and 'video_path' in result:
            # These would normally come from actual quality assessment
            quality_metrics = {
                'visual_quality': 0.85,  # Placeholder
                'temporal_coherence': 0.80,  # Placeholder
                'factual_accuracy': 0.90  # Placeholder
            }
        
        # Track generation end
        monitoring_service.track_generation_end(
            generation_id,
            success=True,
            quality_metrics=quality_metrics
        )
        
        logger.info(f"Generated video for prompt '{prompt}' in {generation_time:.2f}s")
        return result
        
    except Exception as e:
        # Track failed generation
        monitoring_service.track_generation_end(
            generation_id,
            success=False
        )
        
        logger.error(f"Error generating video: {e}")
        raise


def print_system_status(monitoring_service: MonitoringService) -> None:
    """Print the current system status.
    
    Args:
        monitoring_service: MonitoringService instance.
    """
    status = monitoring_service.get_system_status()
    
    print("\n===== SYSTEM STATUS =====")
    print(f"Status: {status['status'].upper()} - {status['status_message']}")
    print("\nSystem Metrics:")
    print(f"  CPU Usage: {status['system_metrics']['cpu_usage']:.1f}%")
    print(f"  Memory Usage: {status['system_metrics']['memory_usage_percent']:.1f}%")
    print(f"  Disk Usage: {status['system_metrics']['disk_usage_percent']:.1f}%")
    print(f"  GPU Usage: {status['system_metrics']['gpu_usage']:.1f}%")
    print(f"  GPU Memory: {status['system_metrics']['gpu_memory_usage']:.1f}%")
    
    print("\nPerformance:")
    print(f"  Total Videos: {status['performance']['total_videos']}")
    print(f"  Success Rate: {status['performance']['success_rate']:.1f}%")
    print(f"  Error Rate: {status['performance']['error_rate']:.1f}%")
    print(f"  Throughput: {status['performance']['throughput']:.2f} videos/min")
    print(f"  Avg Generation Time: {status['performance']['avg_generation_time']:.2f}s")
    print(f"  Active Generations: {status['performance']['active_generations']}")
    
    print("\nAlerts:")
    print(f"  Recent Alerts: {status['alerts']['recent_alerts']}")
    print(f"  Critical: {status['alerts']['critical']}")
    print(f"  Error: {status['alerts']['error']}")
    print(f"  Warning: {status['alerts']['warning']}")
    print(f"  Info: {status['alerts']['info']}")
    print("=======================\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Production monitoring for video generation')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--monitoring-dir', type=str, default='data/monitoring',
                        help='Directory for monitoring data')
    parser.add_argument('--output-dir', type=str, default='data/output',
                        help='Directory for generated videos')
    parser.add_argument('--prompts-file', type=str, default=None,
                        help='JSON file containing prompts to generate videos for')
    parser.add_argument('--status-interval', type=int, default=300,
                        help='Interval in seconds to print system status')
    parser.add_argument('--run-time', type=int, default=3600,
                        help='Total run time in seconds (0 for indefinite)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up monitoring
    monitoring_service = setup_monitoring(args.monitoring_dir)
    
    # Initialize video generator
    generator = VideoGenerator(config_path=args.config)
    
    # Load prompts
    prompts = []
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r') as f:
            prompts_data = json.load(f)
            if isinstance(prompts_data, list):
                prompts = prompts_data
            elif isinstance(prompts_data, dict) and 'prompts' in prompts_data:
                prompts = prompts_data['prompts']
    
    # If no prompts file or empty prompts, use default prompts
    if not prompts:
        prompts = [
            "A policy briefing on renewable energy initiatives",
            "Emergency alert for an incoming hurricane",
            "Instructional guide for workplace safety procedures",
            "Ceremonial announcement for a national holiday"
        ]
    
    logger.info(f"Loaded {len(prompts)} prompts for video generation")
    
    # Print initial system status
    print_system_status(monitoring_service)
    
    # Main loop
    start_time = time.time()
    last_status_time = start_time
    prompt_index = 0
    
    try:
        while True:
            # Check if we've reached the run time limit
            if args.run_time > 0 and time.time() - start_time >= args.run_time:
                logger.info(f"Reached run time limit of {args.run_time} seconds")
                break
            
            # Get the next prompt
            prompt = prompts[prompt_index % len(prompts)]
            prompt_index += 1
            
            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"video_{timestamp}.mp4"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # Generate video with monitoring
            try:
                result = generate_video_with_monitoring(
                    generator,
                    monitoring_service,
                    prompt,
                    output_path
                )
                logger.info(f"Generated video saved to {result.get('video_path', output_path)}")
            except Exception as e:
                logger.error(f"Failed to generate video: {e}")
            
            # Print system status at regular intervals
            current_time = time.time()
            if current_time - last_status_time >= args.status_interval:
                print_system_status(monitoring_service)
                last_status_time = current_time
            
            # Sleep a bit to prevent overloading the system
            time.sleep(5)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    # Print final system status
    print_system_status(monitoring_service)
    
    # Stop monitoring service
    monitoring_service.stop()
    logger.info("Monitoring service stopped")


if __name__ == '__main__':
    main()