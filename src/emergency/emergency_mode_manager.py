import os
import logging
import enum
import json
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmergencyType(enum.Enum):
    """Types of emergency situations."""
    NATURAL_DISASTER = "natural_disaster"
    PUBLIC_SAFETY = "public_safety"
    HEALTH_CRISIS = "health_crisis"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    GENERAL = "general"


class EmergencyPriority(enum.Enum):
    """Priority levels for emergency communications."""
    CRITICAL = "critical"  # Immediate life-threatening situation
    HIGH = "high"          # Urgent situation requiring immediate attention
    MEDIUM = "medium"      # Important but not immediately life-threatening
    LOW = "low"            # Informational emergency communications


@dataclass
class EmergencyModeConfig:
    """Configuration for emergency mode operations."""
    # Generation parameters
    max_generation_time: float = 60.0  # seconds
    reduced_sampling_steps: int = 20   # reduced from normal operation
    reduced_resolution: tuple = (640, 360)  # reduced from normal operation
    
    # Resource allocation
    cpu_priority: int = 10  # higher number = higher priority
    gpu_memory_limit: Optional[float] = None  # GB, None = no limit
    
    # Template settings
    use_templates: bool = True
    template_directory: str = "data/emergency_templates"
    
    # Output settings
    channels: List[str] = field(default_factory=lambda: ["default"])
    accessibility_features: List[str] = field(default_factory=lambda: ["captions", "high_contrast"])
    
    # Compliance settings
    include_disclaimers: bool = True
    compliance_check_level: str = "essential"  # essential, standard, comprehensive
    
    # Logging settings
    detailed_logging: bool = True
    log_directory: str = "logs/emergency"


class EmergencyModeManager:
    """Manager for emergency mode operations in the video generation system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the emergency mode manager.
        
        Args:
            config_path: Path to emergency mode configuration file.
        """
        self.active = False
        self.activation_time = None
        self.config = self._load_config(config_path)
        self.active_channels: Set[str] = set()
        self.activation_history: List[Dict[str, Any]] = []
        self.current_type = None
        self.current_priority = None
        
        # Create log directory if it doesn't exist
        os.makedirs(self.config.log_directory, exist_ok=True)
        
        logger.info("Emergency Mode Manager initialized")
    
    def _load_config(self, config_path: Optional[str]) -> EmergencyModeConfig:
        """Load emergency mode configuration.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            EmergencyModeConfig object.
        """
        if config_path is None or not os.path.exists(config_path):
            logger.info("Using default emergency mode configuration")
            return EmergencyModeConfig()
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create config object with values from file
            config = EmergencyModeConfig(**config_dict)
            logger.info(f"Emergency mode configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading emergency mode configuration: {e}")
            logger.info("Falling back to default emergency mode configuration")
            return EmergencyModeConfig()
    
    def activate(self, 
                emergency_type: EmergencyType = EmergencyType.GENERAL, 
                priority: EmergencyPriority = EmergencyPriority.HIGH,
                channels: Optional[List[str]] = None,
                reason: str = "Manual activation") -> bool:
        """Activate emergency mode.
        
        Args:
            emergency_type: Type of emergency situation.
            priority: Priority level for this emergency.
            channels: List of output channels to activate.
            reason: Reason for activation.
            
        Returns:
            bool: True if activation was successful.
        """
        if self.active:
            logger.warning("Emergency mode already active")
            return True
        
        try:
            self.active = True
            self.activation_time = time.time()
            self.current_type = emergency_type
            self.current_priority = priority
            
            # Set active channels
            if channels:
                self.active_channels = set(channels)
            else:
                self.active_channels = set(self.config.channels)
            
            # Record activation
            activation_record = {
                "timestamp": time.time(),
                "emergency_type": emergency_type.value,
                "priority": priority.value,
                "channels": list(self.active_channels),
                "reason": reason
            }
            self.activation_history.append(activation_record)
            
            # Log activation
            self._log_activation(activation_record)
            
            logger.info(f"Emergency mode activated: {emergency_type.value}, {priority.value}")
            return True
        except Exception as e:
            logger.error(f"Error activating emergency mode: {e}")
            self.active = False
            return False
    
    def deactivate(self, reason: str = "Manual deactivation") -> bool:
        """Deactivate emergency mode.
        
        Args:
            reason: Reason for deactivation.
            
        Returns:
            bool: True if deactivation was successful.
        """
        if not self.active:
            logger.warning("Emergency mode not active")
            return True
        
        try:
            # Calculate duration
            duration = time.time() - self.activation_time if self.activation_time else 0
            
            # Update last activation record
            if self.activation_history:
                self.activation_history[-1]["deactivation_time"] = time.time()
                self.activation_history[-1]["duration"] = duration
                self.activation_history[-1]["deactivation_reason"] = reason
            
            # Log deactivation
            deactivation_record = {
                "timestamp": time.time(),
                "duration": duration,
                "reason": reason
            }
            self._log_deactivation(deactivation_record)
            
            # Reset state
            self.active = False
            self.activation_time = None
            self.active_channels.clear()
            self.current_type = None
            self.current_priority = None
            
            logger.info(f"Emergency mode deactivated after {duration:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Error deactivating emergency mode: {e}")
            return False
    
    def is_active(self) -> bool:
        """Check if emergency mode is active.
        
        Returns:
            bool: True if emergency mode is active.
        """
        return self.active
    
    def get_current_emergency_type(self) -> Optional[EmergencyType]:
        """Get the current emergency type.
        
        Returns:
            Current emergency type, or None if not active.
        """
        return self.current_type if self.active else None
    
    def get_current_priority(self) -> Optional[EmergencyPriority]:
        """Get the current emergency priority.
        
        Returns:
            Current priority, or None if not active.
        """
        return self.current_priority if self.active else None
    
    def get_config_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides for emergency mode.
        
        Returns:
            Dictionary of configuration overrides.
        """
        if not self.active:
            return {}
        
        # Base overrides for video generation
        overrides = {
            "max_generation_time": self.config.max_generation_time,
            "sampling_steps": self.config.reduced_sampling_steps,
            "resolution": self.config.reduced_resolution,
            "use_templates": self.config.use_templates,
            "accessibility_features": self.config.accessibility_features,
            "include_disclaimers": self.config.include_disclaimers
        }
        
        # Add priority-specific overrides if needed
        if self.current_priority == EmergencyPriority.CRITICAL:
            overrides.update({
                "sampling_steps": min(10, self.config.reduced_sampling_steps),  # Further reduce for critical
                "skip_quality_check": True
            })
        
        # Add type-specific overrides if needed
        if self.current_type == EmergencyType.NATURAL_DISASTER:
            overrides.update({
                "include_safety_information": True
            })
        elif self.current_type == EmergencyType.HEALTH_CRISIS:
            overrides.update({
                "include_health_guidelines": True
            })
        
        return overrides
    
    def get_active_channels(self) -> List[str]:
        """Get list of active output channels.
        
        Returns:
            List of active channel names.
        """
        return list(self.active_channels)
    
    def get_activation_history(self) -> List[Dict[str, Any]]:
        """Get history of emergency mode activations.
        
        Returns:
            List of activation records.
        """
        return self.activation_history
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of emergency mode.
        
        Returns:
            Dictionary with status information.
        """
        status = {
            "active": self.active,
            "channels": list(self.active_channels) if self.active else [],
        }
        
        if self.active and self.activation_time:
            status["activation_time"] = self.activation_time
            status["duration"] = time.time() - self.activation_time
            if self.activation_history:
                status["emergency_type"] = self.activation_history[-1]["emergency_type"]
                status["priority"] = self.activation_history[-1]["priority"]
        
        return status
    
    def _log_activation(self, activation_record: Dict[str, Any]):
        """Log emergency mode activation.
        
        Args:
            activation_record: Activation details to log.
        """
        if not self.config.detailed_logging:
            return
        
        try:
            log_file = os.path.join(self.config.log_directory, "emergency_activations.json")
            
            # Load existing log if it exists
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    try:
                        log_data = json.load(f)
                    except json.JSONDecodeError:
                        log_data = {"activations": []}
            else:
                log_data = {"activations": []}
            
            # Add new record
            log_data["activations"].append(activation_record)
            
            # Write updated log
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error logging emergency mode activation: {e}")
    
    def _log_deactivation(self, deactivation_record: Dict[str, Any]):
        """Log emergency mode deactivation.
        
        Args:
            deactivation_record: Deactivation details to log.
        """
        if not self.config.detailed_logging:
            return
        
        try:
            log_file = os.path.join(self.config.log_directory, "emergency_deactivations.json")
            
            # Load existing log if it exists
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    try:
                        log_data = json.load(f)
                    except json.JSONDecodeError:
                        log_data = {"deactivations": []}
            else:
                log_data = {"deactivations": []}
            
            # Add new record
            log_data["deactivations"].append(deactivation_record)
            
            # Write updated log
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error logging emergency mode deactivation: {e}")