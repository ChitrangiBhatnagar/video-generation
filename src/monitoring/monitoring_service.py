"""Monitoring service for the video generation system."""

import os
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable

from .metrics_collector import MetricsCollector, SystemMetrics
from .alert_manager import AlertManager, AlertConfig, AlertLevel
from .performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

class MonitoringService:
    """Integrated monitoring service for the video generation system.
    
    This service combines metrics collection, performance tracking, and alerting
    into a unified monitoring solution for production deployment.
    """
    
    def __init__(self, 
                 base_dir: str = 'data/monitoring',
                 collection_interval: int = 60,  # seconds
                 snapshot_interval: int = 300,  # seconds
                 auto_start: bool = True):
        """Initialize the monitoring service.
        
        Args:
            base_dir: Base directory for monitoring data.
            collection_interval: Interval in seconds for metrics collection.
            snapshot_interval: Interval in seconds for saving performance snapshots.
            auto_start: Whether to automatically start the monitoring service.
        """
        # Create base directory
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize components
        self.metrics_collector = MetricsCollector(metrics_dir=os.path.join(base_dir, 'metrics'))
        self.alert_manager = AlertManager(alerts_dir=os.path.join(base_dir, 'alerts'))
        self.performance_tracker = PerformanceTracker(metrics_dir=os.path.join(base_dir, 'performance'))
        
        # Configuration
        self.collection_interval = collection_interval
        self.snapshot_interval = snapshot_interval
        self.running = False
        self.monitor_thread = None
        self.last_snapshot_time = 0
        
        # Event handlers
        self.on_alert_handlers: List[Callable] = []
        self.on_metrics_update_handlers: List[Callable] = []
        
        # Register default alert handlers
        self._register_default_alert_handlers()
        
        logger.info(f"Initialized monitoring service with base directory at {base_dir}")
        
        # Auto-start if requested
        if auto_start:
            self.start()
    
    def _register_default_alert_handlers(self) -> None:
        """Register default alert handlers."""
        # Log critical alerts
        self.alert_manager.register_alert_handler(
            AlertLevel.CRITICAL,
            lambda alert: logger.critical(f"CRITICAL ALERT: {alert['description']} (value: {alert['actual_value']})")
        )
        
        # Register handler for all alerts to trigger our own handlers
        for level in AlertLevel:
            self.alert_manager.register_alert_handler(
                level,
                self._handle_alert
            )
    
    def _handle_alert(self, alert: Dict[str, Any]) -> None:
        """Handle an alert by notifying registered handlers.
        
        Args:
            alert: Alert dictionary.
        """
        for handler in self.on_alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def register_alert_handler(self, handler: Callable) -> None:
        """Register a handler for all alerts.
        
        Args:
            handler: Callable that takes an alert dict as input.
        """
        self.on_alert_handlers.append(handler)
    
    def register_metrics_update_handler(self, handler: Callable) -> None:
        """Register a handler for metrics updates.
        
        Args:
            handler: Callable that takes a metrics dict as input.
        """
        self.on_metrics_update_handlers.append(handler)
    
    def start(self) -> None:
        """Start the monitoring service."""
        if self.running:
            logger.warning("Monitoring service is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started monitoring service")
    
    def stop(self) -> None:
        """Stop the monitoring service."""
        if not self.running:
            logger.warning("Monitoring service is not running")
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped monitoring service")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                
                # Save metrics
                self.metrics_collector.save_metrics(system_metrics)
                
                # Check for alerts
                combined_metrics = self._get_combined_metrics(system_metrics)
                alerts = self.alert_manager.check_alerts(combined_metrics)
                
                # Save performance snapshot if needed
                current_time = time.time()
                if current_time - self.last_snapshot_time >= self.snapshot_interval:
                    self.performance_tracker.save_performance_snapshot()
                    self.last_snapshot_time = current_time
                
                # Notify metrics update handlers
                for handler in self.on_metrics_update_handlers:
                    try:
                        handler(combined_metrics)
                    except Exception as e:
                        logger.error(f"Error in metrics update handler: {e}")
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Sleep a bit before retrying
    
    def _get_combined_metrics(self, system_metrics: SystemMetrics) -> Dict[str, Any]:
        """Combine all metrics into a single dictionary.
        
        Args:
            system_metrics: Current system metrics.
            
        Returns:
            Combined metrics dictionary.
        """
        # Get performance metrics
        performance_metrics = self.performance_tracker.get_current_metrics()
        
        # Convert SystemMetrics to dict
        system_dict = {
            'cpu_usage': system_metrics.cpu_usage,
            'memory_usage_bytes': system_metrics.memory_usage_bytes,
            'memory_usage_percent': system_metrics.memory_usage_percent,
            'disk_usage_bytes': system_metrics.disk_usage_bytes,
            'disk_usage_percent': system_metrics.disk_usage_percent,
            'network_sent_bytes': system_metrics.network_sent_bytes,
            'network_recv_bytes': system_metrics.network_recv_bytes,
            'gpu_usage': system_metrics.gpu_usage,
            'gpu_memory_usage': system_metrics.gpu_memory_usage,
            'process_count': system_metrics.process_count
        }
        
        # Combine metrics
        combined = {
            'system': system_dict,
            'performance': performance_metrics
        }
        
        # Flatten for easier access in alert conditions
        flattened = {}
        for category, metrics in combined.items():
            for key, value in metrics.items():
                flattened[f"{category}.{key}"] = value
                flattened[key] = value  # Also add without category for backward compatibility
        
        return flattened
    
    def track_generation_start(self, generation_id: str, metadata: Dict[str, Any]) -> None:
        """Track the start of a video generation.
        
        Args:
            generation_id: Unique identifier for the generation.
            metadata: Additional metadata about the generation.
        """
        self.performance_tracker.start_generation(generation_id, metadata)
    
    def track_generation_end(self, generation_id: str, success: bool, quality_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Track the end of a video generation.
        
        Args:
            generation_id: Unique identifier for the generation.
            success: Whether the generation was successful.
            quality_metrics: Optional quality metrics for the generated video.
            
        Returns:
            Dictionary with performance metrics for this generation.
        """
        return self.performance_tracker.end_generation(generation_id, success, quality_metrics)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current system status.
        
        Returns:
            Dictionary containing system status.
        """
        # Get latest metrics
        system_metrics = self.metrics_collector.collect_system_metrics()
        performance_summary = self.performance_tracker.get_performance_summary()
        alert_summary = self.alert_manager.get_alert_summary()
        
        # Determine overall status
        status = 'healthy'
        status_message = 'System is operating normally'
        
        # Check for critical alerts
        if alert_summary.get('level_counts', {}).get('critical', 0) > 0:
            status = 'critical'
            status_message = 'Critical alerts detected'
        elif alert_summary.get('level_counts', {}).get('error', 0) > 0:
            status = 'error'
            status_message = 'Error alerts detected'
        elif alert_summary.get('level_counts', {}).get('warning', 0) > 0:
            status = 'warning'
            status_message = 'Warning alerts detected'
        
        # Check for high resource usage
        if system_metrics.cpu_usage > 95 or system_metrics.memory_usage_percent > 95:
            if status != 'critical':
                status = 'warning'
                status_message = 'High resource usage detected'
        
        # Check for high error rate
        if performance_summary.get('error_rate', 0) > 10:
            if status != 'critical':
                status = 'error'
                status_message = 'High error rate detected'
        
        return {
            'timestamp': time.time(),
            'status': status,
            'status_message': status_message,
            'system_metrics': {
                'cpu_usage': system_metrics.cpu_usage,
                'memory_usage_percent': system_metrics.memory_usage_percent,
                'disk_usage_percent': system_metrics.disk_usage_percent,
                'gpu_usage': system_metrics.gpu_usage,
                'gpu_memory_usage': system_metrics.gpu_memory_usage
            },
            'performance': {
                'total_videos': performance_summary.get('total_videos', 0),
                'success_rate': performance_summary.get('success_rate', 0),
                'error_rate': performance_summary.get('error_rate', 0),
                'throughput': performance_summary.get('throughput', 0),
                'avg_generation_time': performance_summary.get('avg_generation_time', 0),
                'active_generations': performance_summary.get('active_generations', 0)
            },
            'alerts': {
                'recent_alerts': alert_summary.get('recent_alerts', 0),
                'critical': alert_summary.get('level_counts', {}).get('critical', 0),
                'error': alert_summary.get('level_counts', {}).get('error', 0),
                'warning': alert_summary.get('level_counts', {}).get('warning', 0),
                'info': alert_summary.get('level_counts', {}).get('info', 0)
            }
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics for all components.
        
        Returns:
            Dictionary containing detailed metrics.
        """
        return {
            'system': self.metrics_collector.get_latest_metrics(),
            'performance': self.performance_tracker.get_current_metrics(),
            'alerts': self.alert_manager.get_alert_history(limit=10),
            'active_generations': self.performance_tracker.get_active_generations()
        }