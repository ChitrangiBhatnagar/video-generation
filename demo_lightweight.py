"""
Lightweight demo for video generation without heavy model loading.
This creates a simple demonstration video to verify the pipeline works.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import imageio

def create_demo_video(input_image_path, prompt, output_path):
    """
    Create a simple demo video by animating the input image.
    This doesn't use AI models but demonstrates the video pipeline.
    """
    print(f"Creating demo video...")
    print(f"Input: {input_image_path}")
    print(f"Prompt: {prompt}")
    print(f"Output: {output_path}")
    
    # Load input image
    img = Image.open(input_image_path)
    img = img.resize((720, 480))
    img_array = np.array(img)
    
    # Create 49 frames (6 seconds at 8fps)
    frames = []
    num_frames = 49
    
    for i in range(num_frames):
        # Create simple animation effect (zoom + pan)
        progress = i / num_frames
        
        # Zoom effect
        zoom = 1.0 + 0.1 * progress
        h, w = img_array.shape[:2]
        new_h, new_w = int(h / zoom), int(w / zoom)
        
        # Calculate crop
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        
        # Crop and resize
        cropped = img_array[top:top+new_h, left:left+new_w]
        frame = Image.fromarray(cropped).resize((w, h), Image.LANCZOS)
        
        # Add slight brightness variation
        frame_array = np.array(frame).astype(np.float32)
        brightness = 1.0 + 0.1 * np.sin(progress * 3.14159)
        frame_array = np.clip(frame_array * brightness, 0, 255).astype(np.uint8)
        
        frames.append(frame_array)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_frames} frames...")
    
    # Save video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    imageio.mimsave(str(output_path), frames, fps=8)
    print(f"\n✅ Demo video saved to: {output_path}")
    print(f"Duration: 6 seconds")
    print(f"Resolution: 720x480")
    print(f"FPS: 8")
    print(f"Frames: {num_frames}")
    
    return str(output_path)


def main():
    print("=" * 70)
    print("VIDEO GENERATION LIGHTWEIGHT DEMO")
    print("=" * 70)
    print("\nThis demo creates a simple animated video without loading AI models.")
    print("For full AI-powered generation, use cloud GPUs (see README_DETAILED.md)\n")
    
    # Sample images
    samples = [
        ("data/samples/sample_image_1.png", "Press briefing in modern newsroom"),
        ("data/samples/sample_image_2.png", "Market commentary on trading floor"),
        ("data/samples/sample_image_3.png", "Studio interview with expert"),
    ]
    
    print("Available samples:")
    for idx, (img_path, prompt) in enumerate(samples, 1):
        print(f"  {idx}. {img_path} - '{prompt}'")
    
    # Use first sample by default
    input_image = samples[0][0]
    prompt = samples[0][1]
    output_path = "outputs/examples/demo_lightweight.mp4"
    
    if not Path(input_image).exists():
        print(f"\n❌ Error: Sample image not found: {input_image}")
        print("Please ensure data/samples/ directory contains the sample images.")
        return 1
    
    print(f"\n🎬 Generating demo video...")
    print("-" * 70)
    
    try:
        video_path = create_demo_video(input_image, prompt, output_path)
        
        print("\n" + "=" * 70)
        print("SUCCESS! 🎉")
        print("=" * 70)
        print(f"\nYour demo video is ready: {video_path}")
        print("\n💡 Next Steps:")
        print("1. View the video in your media player")
        print("2. For AI-powered generation, use Google Colab or Kaggle")
        print("3. See README_DETAILED.md for cloud GPU setup instructions")
        print("\n🌐 Recommended Cloud Platforms:")
        print("   • Kaggle Notebooks: https://kaggle.com/code (FREE)")
        print("   • Google Colab: https://colab.research.google.com ($10/month)")
        print("   • RunPod: https://runpod.io ($0.20-1.89/hour)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error creating demo video: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
