"""Performance tracking for the video generation system."""

import os
import json
import time
import logging
import datetime
from typing import Dict, List, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks performance metrics for video generation."""
    
    def __init__(self, metrics_dir: str = 'data/monitoring/performance'):
        """Initialize the performance tracker.
        
        Args:
            metrics_dir: Directory to store performance metrics.
        """
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Initialize performance metrics
        self.reset_metrics()
        
        # Keep a rolling window of recent generation times
        self.recent_generation_times = deque(maxlen=100)
        self.recent_quality_scores = deque(maxlen=100)
        
        # Track active generations
        self.active_generations: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized performance tracker with storage at {metrics_dir}")
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.metrics = {
            'total_videos_generated': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_generation_time': 0.0,
            'avg_generation_time': 0.0,
            'min_generation_time': float('inf'),
            'max_generation_time': 0.0,
            'throughput': 0.0,  # videos per minute
            'error_rate': 0.0,
            'avg_visual_quality': 0.0,
            'avg_temporal_coherence': 0.0,
            'avg_factual_accuracy': 0.0,
            'start_time': time.time(),
            'last_update_time': time.time()
        }
    
    def start_generation(self, generation_id: str, metadata: Dict[str, Any]) -> None:
        """Record the start of a video generation.
        
        Args:
            generation_id: Unique identifier for the generation.
            metadata: Additional metadata about the generation.
        """
        self.active_generations[generation_id] = {
            'start_time': time.time(),
            'metadata': metadata
        }
        logger.debug(f"Started tracking generation {generation_id}")
    
    def end_generation(self, generation_id: str, success: bool, quality_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Record the end of a video generation.
        
        Args:
            generation_id: Unique identifier for the generation.
            success: Whether the generation was successful.
            quality_metrics: Optional quality metrics for the generated video.
            
        Returns:
            Dictionary with performance metrics for this generation.
        """
        if generation_id not in self.active_generations:
            logger.warning(f"Generation {generation_id} not found in active generations")
            return {}
        
        generation = self.active_generations.pop(generation_id)
        end_time = time.time()
        generation_time = end_time - generation['start_time']
        
        # Update metrics
        self.metrics['total_videos_generated'] += 1
        if success:
            self.metrics['successful_generations'] += 1
        else:
            self.metrics['failed_generations'] += 1
        
        self.metrics['total_generation_time'] += generation_time
        self.metrics['avg_generation_time'] = self.metrics['total_generation_time'] / self.metrics['total_videos_generated']
        self.metrics['min_generation_time'] = min(self.metrics['min_generation_time'], generation_time)
        self.metrics['max_generation_time'] = max(self.metrics['max_generation_time'], generation_time)
        
        # Calculate error rate
        self.metrics['error_rate'] = self.metrics['failed_generations'] / self.metrics['total_videos_generated'] * 100
        
        # Calculate throughput (videos per minute)
        elapsed_time = end_time - self.metrics['start_time']
        self.metrics['throughput'] = (self.metrics['total_videos_generated'] / elapsed_time) * 60
        
        # Update last update time
        self.metrics['last_update_time'] = end_time
        
        # Add to recent generation times
        self.recent_generation_times.append(generation_time)
        
        # Process quality metrics if provided
        generation_metrics = {
            'generation_id': generation_id,
            'start_time': datetime.datetime.fromtimestamp(generation['start_time']).isoformat(),
            'end_time': datetime.datetime.fromtimestamp(end_time).isoformat(),
            'generation_time': generation_time,
            'success': success,
            'metadata': generation['metadata']
        }
        
        if quality_metrics and success:
            # Update quality metrics
            for metric, value in quality_metrics.items():
                if metric == 'visual_quality':
                    self.recent_quality_scores.append(value)
                    self.metrics['avg_visual_quality'] = sum(self.recent_quality_scores) / len(self.recent_quality_scores)
                elif metric == 'temporal_coherence':
                    self.metrics['avg_temporal_coherence'] = ((self.metrics['avg_temporal_coherence'] * 
                                                             (self.metrics['successful_generations'] - 1)) + value) / \
                                                             self.metrics['successful_generations']
                elif metric == 'factual_accuracy':
                    self.metrics['avg_factual_accuracy'] = ((self.metrics['avg_factual_accuracy'] * 
                                                           (self.metrics['successful_generations'] - 1)) + value) / \
                                                           self.metrics['successful_generations']
            
            generation_metrics['quality_metrics'] = quality_metrics
        
        # Save generation metrics
        self._save_generation_metrics(generation_metrics)
        
        logger.debug(f"Completed tracking generation {generation_id} (time: {generation_time:.2f}s, success: {success})")
        return generation_metrics
    
    def _save_generation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save generation metrics to a file.
        
        Args:
            metrics: Generation metrics dictionary.
        """
        try:
            # Create filename based on date
            date_str = datetime.datetime.now().strftime('%Y%m%d')
            metrics_file = os.path.join(self.metrics_dir, f"generation_metrics_{date_str}.jsonl")
            
            # Append to file
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving generation metrics: {e}")
    
    def save_performance_snapshot(self) -> Dict[str, Any]:
        """Save a snapshot of current performance metrics.
        
        Returns:
            Dictionary containing performance metrics.
        """
        try:
            # Create snapshot with timestamp
            snapshot = self.metrics.copy()
            snapshot['timestamp'] = datetime.datetime.now().isoformat()
            
            # Save to file
            snapshots_file = os.path.join(self.metrics_dir, 'performance_snapshots.jsonl')
            with open(snapshots_file, 'a') as f:
                f.write(json.dumps(snapshot) + '\n')
                
            logger.debug("Saved performance snapshot")
            return snapshot
            
        except Exception as e:
            logger.error(f"Error saving performance snapshot: {e}")
            return {}
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dictionary containing current performance metrics.
        """
        return self.metrics.copy()
    
    def get_recent_generation_times(self) -> List[float]:
        """Get recent generation times.
        
        Returns:
            List of recent generation times.
        """
        return list(self.recent_generation_times)
    
    def get_active_generations(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active generations.
        
        Returns:
            Dictionary of active generations.
        """
        return self.active_generations.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics.
        
        Returns:
            Dictionary containing performance summary.
        """
        if self.metrics['total_videos_generated'] == 0:
            return {
                'status': 'no_data',
                'message': 'No videos have been generated yet'
            }
        
        # Calculate percentiles for generation times
        recent_times = sorted(self.recent_generation_times)
        p50 = p95 = p99 = 0
        
        if recent_times:
            p50 = recent_times[len(recent_times) // 2]
            p95_idx = int(len(recent_times) * 0.95)
            p99_idx = int(len(recent_times) * 0.99)
            p95 = recent_times[p95_idx] if p95_idx < len(recent_times) else recent_times[-1]
            p99 = recent_times[p99_idx] if p99_idx < len(recent_times) else recent_times[-1]
        
        return {
            'status': 'ok',
            'total_videos': self.metrics['total_videos_generated'],
            'success_rate': (self.metrics['successful_generations'] / self.metrics['total_videos_generated']) * 100,
            'error_rate': self.metrics['error_rate'],
            'throughput': self.metrics['throughput'],
            'avg_generation_time': self.metrics['avg_generation_time'],
            'generation_time_p50': p50,
            'generation_time_p95': p95,
            'generation_time_p99': p99,
            'avg_visual_quality': self.metrics['avg_visual_quality'],
            'avg_temporal_coherence': self.metrics['avg_temporal_coherence'],
            'avg_factual_accuracy': self.metrics['avg_factual_accuracy'],
            'active_generations': len(self.active_generations)
        }