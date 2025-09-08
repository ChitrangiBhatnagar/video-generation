import os
import sys
import argparse
import logging

# Add the src directory to the path so we can import the VideoGenerator
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.pib_videogen import VideoGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_directories():
    """Create sample directories for inputs and outputs."""
    # Create sample input directories
    sample_input_dir = os.path.join('data', 'sample_inputs')
    os.makedirs(sample_input_dir, exist_ok=True)
    
    # Create sample output directory
    sample_output_dir = os.path.join('data', 'output_samples')
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Create assets directory for PIB seal
    assets_dir = os.path.join('data', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    logger.info(f"Created sample directories: {sample_input_dir}, {sample_output_dir}, {assets_dir}")
    
    return sample_input_dir, sample_output_dir, assets_dir

def generate_sample_videos(generator, sample_input_dir, collect_feedback=True):
    """Generate sample videos for all use cases.
    
    Args:
        generator: VideoGenerator instance
        sample_input_dir: Directory containing sample inputs
    
    Returns:
        Dictionary mapping use case to generated video path
    """
    # Define sample use cases
    use_cases = {
        "policy_briefing": {
            "text": "The government has announced a new policy to promote renewable energy. The policy aims to increase the share of renewable energy in the total energy mix to 40% by 2030. This will help reduce carbon emissions and combat climate change.",
            "images": [f"{sample_input_dir}/solar_panel.jpg", f"{sample_input_dir}/wind_turbine.jpg"],
            "urgent": False
        },
        "emergency_alert": {
            "text": "URGENT: Heavy rainfall expected in coastal areas. Residents are advised to stay indoors and avoid low-lying areas. Emergency services are on high alert.",
            "images": [f"{sample_input_dir}/heavy_rain.jpg", f"{sample_input_dir}/flood.jpg"],
            "urgent": True
        },
        "instructional_guide": {
            "text": "How to register for COVID-19 vaccination: 1. Visit the CoWIN portal. 2. Register using your mobile number. 3. Book an appointment at your nearest vaccination center. 4. Carry ID proof to the center.",
            "images": [f"{sample_input_dir}/cowin_portal.jpg", f"{sample_input_dir}/vaccination.jpg", f"{sample_input_dir}/id_card.jpg"],
            "urgent": False
        },
        "ceremonial_announcement": {
            "text": "The President of India will be inaugurating the new Parliament building on 28th May 2023. The ceremony will be attended by dignitaries from across the country.",
            "images": [f"{sample_input_dir}/president.jpg", f"{sample_input_dir}/parliament.jpg", f"{sample_input_dir}/ceremony.jpg"],
            "urgent": False
        }
    }
    
    # Generate videos for each use case
    results = {}
    for use_case, params in use_cases.items():
        logger.info(f"Generating video for use case: {use_case}")
        try:
            # Check if sample images exist, if not, use placeholder
            valid_images = []
            for img_path in params["images"]:
                if os.path.exists(img_path):
                    valid_images.append(img_path)
                else:
                    logger.warning(f"Image {img_path} not found. Using placeholder.")
            
            # If no valid images, use placeholder
            if not valid_images:
                logger.warning(f"No valid images for {use_case}. Using placeholder.")
                valid_images = [f"{sample_input_dir}/placeholder.jpg"]
            
            # Generate video
            result = generator.generate_video(
                text=params["text"],
                images=valid_images,
                urgent=params["urgent"]
            )
            
            video_path = result['output_path']
            results[use_case] = video_path
            logger.info(f"Generated video for {use_case}: {video_path}")
            
            # Simulate automated quality metrics for feedback
            if collect_feedback:
                # Simulate automated quality metrics
                quality_metrics = {
                    'visual_quality': round(min(0.7 + 0.3 * (not params["urgent"]), 1.0), 2),  # Urgent videos might have lower quality
                    'temporal_coherence': round(min(0.65 + 0.25 * (not params["urgent"]), 1.0), 2),
                    'factual_accuracy': round(min(0.8 + 0.15 * (not params["urgent"]), 1.0), 2),
                    'generation_time': result['generation_time']
                }
                
                # Record automated metrics feedback
                if hasattr(generator, 'feedback_integration') and generator.feedback_integration:
                    generator.feedback_integration.record_automated_metrics(
                        result['video_id'], 
                        quality_metrics
                    )
                    logger.info(f"Recorded automated metrics for {use_case}")
            
        except Exception as e:
            logger.error(f"Error generating video for {use_case}: {e}")
            results[use_case] = None
    
    return results

def simulate_human_feedback(generator, results):
    """Simulate human feedback for generated videos.
    
    Args:
        generator: VideoGenerator instance
        results: Dictionary mapping use case to generated video path
    """
    logger.info("Simulating human feedback for generated videos")
    
    # Simulate human feedback for each video
    for use_case, video_path in results.items():
        if video_path:
            # Extract video_id from path
            video_id = os.path.basename(video_path).split('.')[0]
            
            # Simulate human feedback scores (slightly higher than automated metrics)
            human_feedback = {
                'visual_quality': round(min(0.75 + 0.25 * (use_case != "emergency_alert"), 1.0), 2),
                'temporal_coherence': round(min(0.7 + 0.3 * (use_case != "emergency_alert"), 1.0), 2),
                'factual_accuracy': round(min(0.85 + 0.15 * (use_case != "emergency_alert"), 1.0), 2),
                'comments': f"Sample human feedback for {use_case} video"
            }
            
            # Record human feedback
            if hasattr(generator, 'feedback_integration') and generator.feedback_integration:
                generator.feedback_integration.record_human_review_feedback(
                    video_id, 
                    human_feedback
                )
                logger.info(f"Recorded human feedback for {use_case}")

def check_for_improvements(generator):
    """Check if model improvements are needed based on feedback data.
    
    Args:
        generator: VideoGenerator instance
    """
    logger.info("Checking for model improvements based on feedback")
    
    if hasattr(generator, 'feedback_integration') and generator.feedback_integration:
        improvement_result = generator.check_for_improvements()
        
        if improvement_result.get('improvements_made', False):
            logger.info("Model improvements were made:")
            for param, change in improvement_result.get('parameter_changes', {}).items():
                logger.info(f"  - {param}: {change['old']} -> {change['new']}")
            
            # Reload config with improvements
            generator.reload_config()
            logger.info("Reloaded generator with improved configuration")
        else:
            logger.info("No model improvements needed at this time")
    else:
        logger.info("Feedback integration not available for improvement checks")

def create_placeholder_images(sample_input_dir):
    """Create placeholder images for sample generation.
    
    In a real implementation, you would use actual images.
    This function creates simple colored rectangles as placeholders.
    
    Args:
        sample_input_dir: Directory to save placeholder images
    """
    try:
        import numpy as np
        import cv2
        
        # Create placeholder images
        placeholders = {
            "placeholder.jpg": (255, 255, 255),  # White
            "solar_panel.jpg": (0, 255, 0),     # Green
            "wind_turbine.jpg": (0, 200, 255),  # Yellow
            "heavy_rain.jpg": (150, 150, 150),  # Gray
            "flood.jpg": (255, 0, 0),           # Blue
            "cowin_portal.jpg": (255, 255, 0),  # Cyan
            "vaccination.jpg": (0, 255, 255),    # Yellow
            "id_card.jpg": (200, 200, 200),     # Light Gray
            "president.jpg": (128, 0, 128),      # Purple
            "parliament.jpg": (165, 42, 42),     # Brown
            "ceremony.jpg": (255, 192, 203)      # Pink
        }
        
        for filename, color in placeholders.items():
            # Create a colored image
            img = np.ones((512, 512, 3), dtype=np.uint8)
            img[:, :] = color
            
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = filename.split('.')[0]
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            
            # Get coordinates to center the text
            textX = (img.shape[1] - textsize[0]) // 2
            textY = (img.shape[0] + textsize[1]) // 2
            
            # Add text to image
            cv2.putText(img, text, (textX, textY), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Save image
            filepath = os.path.join(sample_input_dir, filename)
            cv2.imwrite(filepath, img)
            logger.info(f"Created placeholder image: {filepath}")
    
    except ImportError:
        logger.error("Could not import numpy or cv2. Placeholder images will not be created.")
    except Exception as e:
        logger.error(f"Error creating placeholder images: {e}")

def create_pib_seal(assets_dir):
    """Create a placeholder PIB seal.
    
    In a real implementation, you would use the actual PIB seal.
    This function creates a simple placeholder.
    
    Args:
        assets_dir: Directory to save the PIB seal
    """
    try:
        import numpy as np
        import cv2
        
        # Create a white image
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Draw a circle for the seal
        cv2.circle(img, (100, 100), 80, (0, 0, 128), 5)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "PIB", (70, 110), font, 1.5, (0, 0, 128), 2, cv2.LINE_AA)
        
        # Save image
        filepath = os.path.join(assets_dir, "pib_seal.png")
        cv2.imwrite(filepath, img)
        logger.info(f"Created placeholder PIB seal: {filepath}")
    
    except ImportError:
        logger.error("Could not import numpy or cv2. PIB seal will not be created.")
    except Exception as e:
        logger.error(f"Error creating PIB seal: {e}")

def main():
    """Main function to generate sample videos for all use cases."""
    parser = argparse.ArgumentParser(description="Generate sample videos for PIB-VideoGen")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--feedback", action="store_true", help="Enable feedback loop demonstration")
    parser.add_argument("--iterations", type=int, default=2, help="Number of generation iterations for feedback demo")
    args = parser.parse_args()
    
    try:
        # Create sample directories
        sample_input_dir, sample_output_dir, assets_dir = create_sample_directories()
        
        # Create placeholder images and PIB seal
        create_placeholder_images(sample_input_dir)
        create_pib_seal(assets_dir)
        
        # Initialize the video generator
        generator = VideoGenerator(args.config)
        
        if args.feedback:
            logger.info(f"Running feedback loop demonstration with {args.iterations} iterations")
            
            for iteration in range(args.iterations):
                logger.info(f"\n=== Iteration {iteration+1}/{args.iterations} ===")
                
                # Generate sample videos
                results = generate_sample_videos(generator, sample_input_dir, collect_feedback=True)
                
                # Simulate human feedback
                simulate_human_feedback(generator, results)
                
                # Check for improvements
                check_for_improvements(generator)
                
                # Get feedback summary
                if hasattr(generator, 'feedback_integration') and generator.feedback_integration:
                    summary = generator.get_feedback_summary()
                    logger.info(f"Feedback Summary: {summary}")
                
                # Print iteration summary
                print(f"\nIteration {iteration+1} Summary:")
                for use_case, video_path in results.items():
                    status = "SUCCESS" if video_path else "FAILED"
                    print(f"{use_case}: {status}")
                
                # Get improvement history
                if iteration == args.iterations - 1 and hasattr(generator, 'feedback_integration'):
                    history = generator.get_improvement_history()
                    if history:
                        print("\nImprovement History:")
                        for entry in history:
                            print(f"  - {entry['timestamp']}: {entry['description']}")
                    else:
                        print("\nNo improvements were made during this run.")
        else:
            # Generate sample videos without feedback loop
            results = generate_sample_videos(generator, sample_input_dir, collect_feedback=False)
            
            # Print summary
            print("\nSample Video Generation Summary:")
            print("-" * 40)
            for use_case, video_path in results.items():
                status = "SUCCESS" if video_path else "FAILED"
                print(f"{use_case.upper()}: {status}")
                if video_path:
                    print(f"  - Video: {video_path}")
            print("-" * 40)
            
            # Count successes
            success_count = sum(1 for path in results.values() if path)
            print(f"Successfully generated {success_count}/{len(results)} sample videos.")
            
            if success_count == len(results):
                print("\nAll sample videos generated successfully!")
            else:
                print("\nSome sample videos failed to generate. Check the logs for details.")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())