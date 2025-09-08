"""Metrics collector for the video generation system."""

import os
import time
import psutil
import logging
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System metrics for monitoring."""
    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0  # In MB
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0  # In MB
    disk_usage: float = 0.0  # In MB
    
    # Performance metrics
    avg_generation_time: float = 0.0  # In seconds
    throughput: float = 0.0  # Videos per minute
    queue_size: int = 0
    active_generations: int = 0
    
    # Quality metrics
    avg_visual_quality: float = 0.0
    avg_temporal_coherence: float = 0.0
    avg_factual_accuracy: float = 0.0
    
    # Error metrics
    error_count: int = 0
    error_rate: float = 0.0  # Errors per 100 generations
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


class MetricsCollector:
    """Collects system metrics for monitoring."""
    
    def __init__(self, metrics_dir: str = 'data/monitoring/metrics'):
        """Initialize the metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics data.
        """
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Initialize metrics history
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 1000  # Keep last 1000 metrics points
        
        # Generation tracking
        self.generation_times: List[float] = []
        self.generation_start_times: Dict[str, float] = {}
        self.active_generations: Dict[str, Dict[str, Any]] = {}
        self.completed_generations: int = 0
        self.failed_generations: int = 0
        
        # Quality tracking
        self.quality_metrics: List[Dict[str, float]] = []
        
        logger.info(f"Initialized metrics collector with storage at {metrics_dir}")
    
    def start_generation_tracking(self, video_id: str) -> None:
        """Start tracking a video generation.
        
        Args:
            video_id: Unique identifier for the video.
        """
        self.generation_start_times[video_id] = time.time()
        self.active_generations[video_id] = {
            'start_time': time.time(),
            'status': 'in_progress'
        }
        logger.debug(f"Started tracking generation for video {video_id}")
    
    def end_generation_tracking(self, video_id: str, success: bool = True, 
                               quality_metrics: Optional[Dict[str, float]] = None) -> None:
        """End tracking a video generation.
        
        Args:
            video_id: Unique identifier for the video.
            success: Whether the generation was successful.
            quality_metrics: Quality metrics for the generated video.
        """
        if video_id in self.generation_start_times:
            generation_time = time.time() - self.generation_start_times[video_id]
            self.generation_times.append(generation_time)
            
            # Keep only the last 100 generation times
            if len(self.generation_times) > 100:
                self.generation_times.pop(0)
            
            # Update generation counts
            if success:
                self.completed_generations += 1
            else:
                self.failed_generations += 1
            
            # Remove from active generations
            if video_id in self.active_generations:
                self.active_generations.pop(video_id)
            
            # Remove from start times
            self.generation_start_times.pop(video_id)
            
            # Record quality metrics if provided
            if quality_metrics:
                self.quality_metrics.append(quality_metrics)
                # Keep only the last 100 quality metrics
                if len(self.quality_metrics) > 100:
                    self.quality_metrics.pop(0)
            
            logger.debug(f"Ended tracking generation for video {video_id} (success={success})")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics.
        
        Returns:
            SystemMetrics object with current metrics.
        """
        try:
            # Collect resource usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB
            disk_usage = psutil.disk_usage('/').used / (1024 * 1024)  # Convert to MB
            
            # GPU metrics would require additional libraries like pynvml
            # For this example, we'll use placeholder values
            gpu_usage = 0.0
            gpu_memory_usage = 0.0
            
            # Calculate performance metrics
            avg_generation_time = sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0.0
            throughput = len(self.generation_times) / (sum(self.generation_times) / 60) if self.generation_times else 0.0
            
            # Calculate quality metrics
            avg_visual_quality = sum(m.get('visual_quality', 0) for m in self.quality_metrics) / len(self.quality_metrics) if self.quality_metrics else 0.0
            avg_temporal_coherence = sum(m.get('temporal_coherence', 0) for m in self.quality_metrics) / len(self.quality_metrics) if self.quality_metrics else 0.0
            avg_factual_accuracy = sum(m.get('factual_accuracy', 0) for m in self.quality_metrics) / len(self.quality_metrics) if self.quality_metrics else 0.0
            
            # Calculate error metrics
            total_generations = self.completed_generations + self.failed_generations
            error_rate = (self.failed_generations / total_generations * 100) if total_generations > 0 else 0.0
            
            # Create metrics object
            metrics = SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                gpu_memory_usage=gpu_memory_usage,
                disk_usage=disk_usage,
                avg_generation_time=avg_generation_time,
                throughput=throughput,
                queue_size=0,  # Would be provided by a queue manager
                active_generations=len(self.active_generations),
                avg_visual_quality=avg_visual_quality,
                avg_temporal_coherence=avg_temporal_coherence,
                avg_factual_accuracy=avg_factual_accuracy,
                error_count=self.failed_generations,
                error_rate=error_rate
            )
            
            # Add to history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            # Save metrics to file
            self._save_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics()
    
    def _save_metrics(self, metrics: SystemMetrics) -> None:
        """Save metrics to a file.
        
        Args:
            metrics: SystemMetrics object to save.
        """
        try:
            # Create filename based on date
            date_str = datetime.datetime.now().strftime('%Y%m%d')
            metrics_file = os.path.join(self.metrics_dir, f"metrics_{date_str}.jsonl")
            
            # Convert metrics to dict
            metrics_dict = {
                'timestamp': metrics.timestamp,
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'gpu_usage': metrics.gpu_usage,
                'gpu_memory_usage': metrics.gpu_memory_usage,
                'disk_usage': metrics.disk_usage,
                'avg_generation_time': metrics.avg_generation_time,
                'throughput': metrics.throughput,
                'queue_size': metrics.queue_size,
                'active_generations': metrics.active_generations,
                'avg_visual_quality': metrics.avg_visual_quality,
                'avg_temporal_coherence': metrics.avg_temporal_coherence,
                'avg_factual_accuracy': metrics.avg_factual_accuracy,
                'error_count': metrics.error_count,
                'error_rate': metrics.error_rate
            }
            
            # Append to file
            import json
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics_dict) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def get_metrics_history(self, limit: int = 100) -> List[SystemMetrics]:
        """Get metrics history.
        
        Args:
            limit: Maximum number of metrics points to return.
            
        Returns:
            List of SystemMetrics objects.
        """
        return self.metrics_history[-limit:] if self.metrics_history else []
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics.
        
        Returns:
            SystemMetrics object with current metrics.
        """
        return self.collect_system_metrics()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of system metrics.
        
        Returns:
            Dictionary containing metrics summary.
        """
        if not self.metrics_history:
            return {
                'status': 'no_data',
                'message': 'No metrics data available'
            }
        
        # Get the latest metrics
        latest = self.metrics_history[-1]
        
        # Calculate averages over the last hour (assuming metrics are collected every minute)
        hour_metrics = self.metrics_history[-60:] if len(self.metrics_history) >= 60 else self.metrics_history
        
        avg_cpu = sum(m.cpu_usage for m in hour_metrics) / len(hour_metrics)
        avg_memory = sum(m.memory_usage for m in hour_metrics) / len(hour_metrics)
        avg_generation_time = sum(m.avg_generation_time for m in hour_metrics) / len(hour_metrics)
        avg_throughput = sum(m.throughput for m in hour_metrics) / len(hour_metrics)
        avg_error_rate = sum(m.error_rate for m in hour_metrics) / len(hour_metrics)
        
        return {
            'status': 'ok',
            'timestamp': latest.timestamp,
            'current': {
                'cpu_usage': latest.cpu_usage,
                'memory_usage': latest.memory_usage,
                'gpu_usage': latest.gpu_usage,
                'avg_generation_time': latest.avg_generation_time,
                'throughput': latest.throughput,
                'active_generations': latest.active_generations,
                'error_rate': latest.error_rate
            },
            'hourly_average': {
                'cpu_usage': avg_cpu,
                'memory_usage': avg_memory,
                'avg_generation_time': avg_generation_time,
                'throughput': avg_throughput,
                'error_rate': avg_error_rate
            },
            'total_generations': self.completed_generations + self.failed_generations,
            'successful_generations': self.completed_generations,
            'failed_generations': self.failed_generations
        }