import os
import logging
import datetime
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeedbackIntegration:
    """Integrates feedback loop with the main VideoGenerator system."""
    
    def __init__(self, video_generator=None):
        """Initialize the FeedbackIntegration.
        
        Args:
            video_generator: Reference to the VideoGenerator instance.
        """
        self.video_generator = video_generator
        
        # Import here to avoid circular imports
        from .feedback_collector import FeedbackCollector
        from .model_improver import ModelImprover
        
        self.feedback_collector = FeedbackCollector()
        self.model_improver = ModelImprover()
        
        # Set up automatic improvement schedule
        self.auto_improvement_enabled = True
        
        logger.info("FeedbackIntegration initialized")
    
    def register_video_generator(self, video_generator):
        """Register the VideoGenerator instance.
        
        Args:
            video_generator: Reference to the VideoGenerator instance.
        """
        self.video_generator = video_generator
        logger.info("VideoGenerator registered with FeedbackIntegration")
    
    def collect_automated_metrics(self, video_id: str, video_data: Dict[str, Any]) -> Dict[str, float]:
        """Collect automated metrics for a generated video.
        
        Args:
            video_id: ID of the generated video.
            video_data: Video data and metadata.
            
        Returns:
            Dictionary of automated metrics.
        """
        # This would normally call into the quality_compliance module
        # For now, we'll return placeholder metrics
        return {
            'fvd_score': 180.0,  # Lower is better
            'is_score': 75.0,     # Higher is better
            'temporal_consistency': 0.85,  # Higher is better
            'artifact_count': 2,   # Lower is better
            'generation_time': video_data.get('generation_time', 0.0)
        }
    
    def record_generation_feedback(self, video_id: str, video_data: Dict[str, Any]) -> bool:
        """Record automated feedback for a newly generated video.
        
        Args:
            video_id: ID of the generated video.
            video_data: Video data and metadata.
            
        Returns:
            bool: True if feedback was successfully recorded.
        """
        if not self.feedback_collector:
            logger.error("FeedbackCollector not initialized")
            return False
        
        try:
            # Collect automated metrics
            automated_metrics = self.collect_automated_metrics(video_id, video_data)
            
            # Import here to avoid circular imports
            from .feedback_collector import FeedbackMetrics
            
            # Create feedback metrics
            feedback = FeedbackMetrics(
                video_id=video_id,
                timestamp=datetime.datetime.now().isoformat(),
                generation_time=video_data.get('generation_time', 0.0),
                visual_quality_score=8.0,  # Placeholder - would be calculated from metrics
                temporal_coherence_score=automated_metrics['temporal_consistency'] * 10,
                factual_accuracy_score=9.0,  # Placeholder - would be calculated from fact check results
                automated_metrics=automated_metrics
            )
            
            # Record feedback
            return self.feedback_collector.record_feedback(feedback)
        
        except Exception as e:
            logger.error(f"Error recording generation feedback: {e}")
            return False
    
    def record_human_review_feedback(self, video_id: str, reviewer_id: str, 
                                     scores: Dict[str, float], notes: str = "") -> bool:
        """Record human review feedback for a video.
        
        Args:
            video_id: ID of the reviewed video.
            reviewer_id: ID of the human reviewer.
            scores: Dictionary of scores (visual_quality, temporal_coherence, factual_accuracy).
            notes: Review notes.
            
        Returns:
            bool: True if feedback was successfully recorded.
        """
        if not self.feedback_collector:
            logger.error("FeedbackCollector not initialized")
            return False
        
        try:
            # Get existing feedback for this video
            recent_feedback = self.feedback_collector.get_recent_feedback(100)
            existing_feedback = next((f for f in recent_feedback if f['video_id'] == video_id), None)
            
            if not existing_feedback:
                # Create new feedback entry if none exists
                from .feedback_collector import FeedbackMetrics
                
                feedback = FeedbackMetrics(
                    video_id=video_id,
                    timestamp=datetime.datetime.now().isoformat(),
                    generation_time=0.0,  # Unknown
                    visual_quality_score=scores.get('visual_quality', 0.0),
                    temporal_coherence_score=scores.get('temporal_coherence', 0.0),
                    factual_accuracy_score=scores.get('factual_accuracy', 0.0),
                    human_reviewer_id=reviewer_id,
                    human_review_notes=notes
                )
                
                return self.feedback_collector.record_feedback(feedback)
            else:
                # Update existing feedback with human review
                from .feedback_collector import FeedbackMetrics
                
                feedback = FeedbackMetrics(
                    video_id=video_id,
                    timestamp=existing_feedback['timestamp'],
                    generation_time=existing_feedback['generation_time'],
                    visual_quality_score=scores.get('visual_quality', existing_feedback['visual_quality_score']),
                    temporal_coherence_score=scores.get('temporal_coherence', existing_feedback['temporal_coherence_score']),
                    factual_accuracy_score=scores.get('factual_accuracy', existing_feedback['factual_accuracy_score']),
                    human_reviewer_id=reviewer_id,
                    human_review_notes=notes,
                    automated_metrics=existing_feedback.get('automated_metrics'),
                    user_feedback=existing_feedback.get('user_feedback')
                )
                
                return self.feedback_collector.record_feedback(feedback)
        
        except Exception as e:
            logger.error(f"Error recording human review feedback: {e}")
            return False
    
    def check_for_improvements(self) -> Dict[str, Any]:
        """Check if model improvements are needed based on feedback data.
        
        Returns:
            Dictionary containing improvement status and details.
        """
        if not self.model_improver:
            logger.error("ModelImprover not initialized")
            return {'status': 'error', 'message': 'ModelImprover not initialized'}
        
        # Check for improvements
        improvement_result = self.model_improver.check_for_improvements()
        
        # If improvements were applied, notify the VideoGenerator to reload config
        if improvement_result.get('status') == 'improvements_applied' and self.video_generator:
            try:
                # This assumes VideoGenerator has a reload_config method
                self.video_generator.reload_config()
                improvement_result['video_generator_updated'] = True
            except Exception as e:
                logger.error(f"Error updating VideoGenerator config: {e}")
                improvement_result['video_generator_updated'] = False
        
        return improvement_result
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get a summary of recent feedback.
        
        Returns:
            Dictionary containing feedback summary.
        """
        if not self.feedback_collector:
            logger.error("FeedbackCollector not initialized")
            return {'status': 'error', 'message': 'FeedbackCollector not initialized'}
        
        return self.feedback_collector.calculate_metrics_summary()
    
    def get_improvement_history(self) -> List[Dict[str, Any]]:
        """Get the history of model improvements.
        
        Returns:
            List of improvement entries.
        """
        improvement_logs_path = os.path.join('data', 'improvement_logs')
        improvement_history = []
        
        if not os.path.exists(improvement_logs_path):
            return []
        
        try:
            # Get all improvement log files
            log_files = [f for f in os.listdir(improvement_logs_path) 
                        if f.startswith('improvement_log_') and f.endswith('.json')]
            
            # Load each log file
            for log_file in log_files:
                try:
                    with open(os.path.join(improvement_logs_path, log_file), 'r') as f:
                        import json
                        log_data = json.load(f)
                        improvement_history.append(log_data)
                except Exception as e:
                    logger.error(f"Error loading improvement log {log_file}: {e}")
            
            # Sort by timestamp (newest first)
            improvement_history.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return improvement_history
        
        except Exception as e:
            logger.error(f"Error getting improvement history: {e}")
            return []
            
    def get_latest_improved_config(self) -> Optional[Any]:
        """Get the latest improved configuration if available.
        
        Returns:
            The latest improved configuration or None if no improvements have been made.
        """
        return self.model_improver.get_latest_improved_config()