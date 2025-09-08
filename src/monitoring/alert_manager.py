"""Alert manager for the video generation system."""

import os
import json
import logging
import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AlertConfig:
    """Configuration for an alert."""
    name: str
    description: str
    level: AlertLevel
    threshold: float
    metric_path: str  # Path to the metric in the metrics object (e.g., 'cpu_usage')
    comparison: str  # 'gt' (greater than), 'lt' (less than), 'eq' (equal)
    enabled: bool = True

class AlertManager:
    """Manages alerts for the monitoring system."""
    
    def __init__(self, alerts_dir: str = 'data/monitoring/alerts'):
        """Initialize the alert manager.
        
        Args:
            alerts_dir: Directory to store alert data.
        """
        self.alerts_dir = alerts_dir
        os.makedirs(alerts_dir, exist_ok=True)
        
        # Initialize alert configurations
        self.alert_configs: List[AlertConfig] = self._load_default_alert_configs()
        
        # Initialize alert history
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000  # Keep last 1000 alerts
        
        # Alert handlers
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = {
            AlertLevel.INFO: [],
            AlertLevel.WARNING: [],
            AlertLevel.ERROR: [],
            AlertLevel.CRITICAL: []
        }
        
        logger.info(f"Initialized alert manager with storage at {alerts_dir}")
    
    def _load_default_alert_configs(self) -> List[AlertConfig]:
        """Load default alert configurations.
        
        Returns:
            List of AlertConfig objects.
        """
        return [
            AlertConfig(
                name="high_cpu_usage",
                description="CPU usage is above 90%",
                level=AlertLevel.WARNING,
                threshold=90.0,
                metric_path="cpu_usage",
                comparison="gt"
            ),
            AlertConfig(
                name="high_memory_usage",
                description="Memory usage is above 90%",
                level=AlertLevel.WARNING,
                threshold=90.0,
                metric_path="memory_usage_percent",
                comparison="gt"
            ),
            AlertConfig(
                name="high_gpu_usage",
                description="GPU usage is above 95%",
                level=AlertLevel.WARNING,
                threshold=95.0,
                metric_path="gpu_usage",
                comparison="gt"
            ),
            AlertConfig(
                name="high_error_rate",
                description="Error rate is above 5%",
                level=AlertLevel.ERROR,
                threshold=5.0,
                metric_path="error_rate",
                comparison="gt"
            ),
            AlertConfig(
                name="low_throughput",
                description="Throughput is below 1 video per minute",
                level=AlertLevel.WARNING,
                threshold=1.0,
                metric_path="throughput",
                comparison="lt"
            ),
            AlertConfig(
                name="high_generation_time",
                description="Average generation time is above 60 seconds",
                level=AlertLevel.WARNING,
                threshold=60.0,
                metric_path="avg_generation_time",
                comparison="gt"
            ),
            AlertConfig(
                name="low_visual_quality",
                description="Average visual quality is below 0.7",
                level=AlertLevel.WARNING,
                threshold=0.7,
                metric_path="avg_visual_quality",
                comparison="lt"
            ),
            AlertConfig(
                name="low_factual_accuracy",
                description="Average factual accuracy is below 0.8",
                level=AlertLevel.ERROR,
                threshold=0.8,
                metric_path="avg_factual_accuracy",
                comparison="lt"
            )
        ]
    
    def register_alert_handler(self, level: AlertLevel, handler: Callable) -> None:
        """Register a handler for alerts of a specific level.
        
        Args:
            level: Alert level to handle.
            handler: Callable that takes an alert dict as input.
        """
        self.alert_handlers[level].append(handler)
        logger.debug(f"Registered alert handler for level {level.value}")
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if any alerts should be triggered based on current metrics.
        
        Args:
            metrics: Current system metrics.
            
        Returns:
            List of triggered alerts.
        """
        triggered_alerts = []
        
        for config in self.alert_configs:
            if not config.enabled:
                continue
                
            # Get the metric value
            metric_value = self._get_metric_value(metrics, config.metric_path)
            if metric_value is None:
                continue
                
            # Check if the alert should be triggered
            triggered = False
            if config.comparison == 'gt' and metric_value > config.threshold:
                triggered = True
            elif config.comparison == 'lt' and metric_value < config.threshold:
                triggered = True
            elif config.comparison == 'eq' and metric_value == config.threshold:
                triggered = True
                
            if triggered:
                alert = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'name': config.name,
                    'description': config.description,
                    'level': config.level.value,
                    'threshold': config.threshold,
                    'actual_value': metric_value,
                    'metric_path': config.metric_path
                }
                
                triggered_alerts.append(alert)
                self._handle_alert(alert)
                
        return triggered_alerts
    
    def _get_metric_value(self, metrics: Dict[str, Any], metric_path: str) -> Optional[float]:
        """Get a metric value from the metrics dictionary.
        
        Args:
            metrics: Metrics dictionary.
            metric_path: Path to the metric (e.g., 'cpu_usage').
            
        Returns:
            Metric value or None if not found.
        """
        try:
            # Split the path into parts
            parts = metric_path.split('.')
            
            # Navigate to the metric
            value = metrics
            for part in parts:
                value = value.get(part)
                if value is None:
                    return None
                    
            return float(value)
        except (KeyError, TypeError, ValueError):
            return None
    
    def _handle_alert(self, alert: Dict[str, Any]) -> None:
        """Handle a triggered alert.
        
        Args:
            alert: Alert dictionary.
        """
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history_size:
            self.alert_history.pop(0)
            
        # Save to file
        self._save_alert(alert)
        
        # Log the alert
        level_str = alert['level'].upper()
        logger.warning(f"ALERT [{level_str}]: {alert['description']} (value: {alert['actual_value']}, threshold: {alert['threshold']})")
        
        # Call handlers
        level = AlertLevel(alert['level'])
        for handler in self.alert_handlers[level]:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def _save_alert(self, alert: Dict[str, Any]) -> None:
        """Save an alert to a file.
        
        Args:
            alert: Alert dictionary.
        """
        try:
            # Create filename based on date
            date_str = datetime.datetime.now().strftime('%Y%m%d')
            alerts_file = os.path.join(self.alerts_dir, f"alerts_{date_str}.jsonl")
            
            # Append to file
            with open(alerts_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
    
    def get_alert_history(self, level: Optional[AlertLevel] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history.
        
        Args:
            level: Filter by alert level.
            limit: Maximum number of alerts to return.
            
        Returns:
            List of alert dictionaries.
        """
        if level:
            filtered = [a for a in self.alert_history if a['level'] == level.value]
            return filtered[-limit:] if filtered else []
        else:
            return self.alert_history[-limit:] if self.alert_history else []
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get a summary of recent alerts.
        
        Returns:
            Dictionary containing alert summary.
        """
        if not self.alert_history:
            return {
                'status': 'no_alerts',
                'message': 'No alerts have been triggered'
            }
        
        # Count alerts by level in the last hour
        hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
        hour_ago_str = hour_ago.isoformat()
        
        recent_alerts = [a for a in self.alert_history if a['timestamp'] > hour_ago_str]
        
        level_counts = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'critical': 0
        }
        
        for alert in recent_alerts:
            level_counts[alert['level']] += 1
        
        # Get the most recent alert
        most_recent = self.alert_history[-1] if self.alert_history else None
        
        return {
            'status': 'ok',
            'total_alerts': len(self.alert_history),
            'recent_alerts': len(recent_alerts),
            'level_counts': level_counts,
            'most_recent': most_recent
        }