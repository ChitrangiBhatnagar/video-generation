#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Emergency Mode Example

This example demonstrates how to use the emergency mode capabilities
of the video generation system for time-sensitive content creation.
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pib_videogen import VideoGenerator, VideoGenConfig
from src.emergency import EmergencyType, EmergencyPriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the emergency mode example."""
    # Create a configuration with emergency mode enabled
    config = VideoGenConfig(
        # Basic configuration
        model_path="models/video_diffusion",
        output_path="output/videos",
        resolution=(1280, 720),
        fps=30,
        
        # Emergency mode configuration
        emergency_mode_enabled=True,
        emergency_reduced_resolution=(640, 360),
        emergency_reduced_sampling_steps=20,
        emergency_template_directory="templates/emergency",
        emergency_use_templates=True,
        emergency_channels=["broadcast", "mobile", "social_media"],
        emergency_accessibility_features=["high_contrast", "large_text"],
        emergency_include_disclaimers=True
    )
    
    # Initialize the video generator
    generator = VideoGenerator(config)
    
    # Example 1: Generate a video with emergency mode explicitly enabled
    logger.info("Example 1: Generating video with emergency mode explicitly enabled")
    emergency_video_path = generator.generate(
        text="Flash flood warning for Riverside County. Evacuate low-lying areas immediately.",
        images=["examples/images/flood1.jpg", "examples/images/flood2.jpg"],
        emergency_mode=True,
        emergency_type="NATURAL_DISASTER",
        emergency_priority="CRITICAL",
        channels=["broadcast", "mobile", "emergency_broadcast_system"]
    )
    logger.info(f"Emergency video generated: {emergency_video_path}")
    
    # Example 2: Generate a video with the urgent flag (simplified emergency mode)
    logger.info("Example 2: Generating video with urgent flag")
    urgent_video_path = generator.generate(
        text="Police activity in downtown area. Avoid 5th and Main Street until further notice.",
        images=["examples/images/police1.jpg"],
        urgent=True
    )
    logger.info(f"Urgent video generated: {urgent_video_path}")
    
    # Example 3: Activate global emergency mode and then generate videos
    logger.info("Example 3: Activating global emergency mode")
    if generator.emergency_mode_manager:
        # Activate emergency mode globally
        activation_success = generator.emergency_mode_manager.activate(
            emergency_type=EmergencyType.HEALTH_ALERT,
            priority=EmergencyPriority.HIGH,
            reason="Demonstration of global emergency mode"
        )
        
        if activation_success:
            logger.info("Global emergency mode activated successfully")
            
            # Generate a video - emergency mode will be automatically applied
            # even though we don't explicitly set emergency_mode=True
            auto_emergency_video_path = generator.generate(
                text="Health alert: Air quality warning in effect. Limit outdoor activities.",
                images=["examples/images/air_quality.jpg"]
            )
            logger.info(f"Auto-emergency video generated: {auto_emergency_video_path}")
            
            # Deactivate emergency mode
            generator.emergency_mode_manager.deactivate(
                reason="Demonstration completed"
            )
            logger.info("Global emergency mode deactivated")
        else:
            logger.error("Failed to activate global emergency mode")
    else:
        logger.warning("Emergency mode manager not available")
    
    # Example 4: Using a specific template
    logger.info("Example 4: Using a specific template for emergency content")
    if generator.emergency_template_library:
        # Find templates for a specific emergency type
        templates = generator.emergency_template_library.find_templates(
            emergency_type="PUBLIC_SAFETY",
            priority_level="HIGH"
        )
        
        if templates:
            template = templates[0]
            logger.info(f"Found template: {template.name}")
            
            # Format the template with variables
            variables = {
                "message": "Hazardous materials spill on Highway 101. Road closed between exits 25-30.",
                "timestamp": "2023-07-20 15:30:00"
            }
            
            formatted_prompt = template.format_prompt(variables)
            logger.info(f"Formatted prompt: {formatted_prompt}")
            
            # Generate video using the formatted prompt
            template_video_path = generator.generate(
                text=formatted_prompt,
                images=["examples/images/hazmat.jpg"],
                emergency_mode=True,
                emergency_type="PUBLIC_SAFETY",
                emergency_priority="HIGH"
            )
            logger.info(f"Template-based video generated: {template_video_path}")
        else:
            logger.warning("No matching templates found")
    else:
        logger.warning("Emergency template library not available")
    
    logger.info("Emergency mode examples completed")


if __name__ == "__main__":
    main()