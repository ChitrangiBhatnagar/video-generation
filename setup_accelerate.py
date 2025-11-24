"""
Quick setup script for accelerate configuration.
Run this once to configure accelerate for your system.
"""

import subprocess
import sys
from pathlib import Path

def setup_accelerate():
    """Configure accelerate with sensible defaults."""
    
    print("=" * 60)
    print("Setting up Accelerate Configuration")
    print("=" * 60)
    
    # Create accelerate config directory if it doesn't exist
    config_dir = Path.home() / ".cache" / "huggingface" / "accelerate"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "default_config.yaml"
    
    # Basic config for single GPU or CPU
    config_content = """compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"\n✅ Accelerate config created at: {config_file}")
    print("\nConfiguration:")
    print("  - Compute Environment: LOCAL_MACHINE")
    print("  - Distributed Type: NO (single device)")
    print("  - Mixed Precision: NO (full precision)")
    print("  - Num Processes: 1")
    print("\n" + "=" * 60)
    print("Setup complete! You can now run training without warnings.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        setup_accelerate()
        print("\n✨ To verify configuration, run:")
        print("   accelerate env")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("\nYou can manually configure by running:")
        print("   accelerate config")
        sys.exit(1)
