import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FeedbackMetrics:
    """Metrics collected from video generation feedback."""
    video_id: str
    timestamp: str
    generation_time: float  # in seconds
    visual_quality_score: float  # 0-10 scale
    temporal_coherence_score: float  # 0-10 scale
    factual_accuracy_score: float  # 0-10 scale
    human_reviewer_id: Optional[str] = None
    human_review_notes: Optional[str] = None
    automated_metrics: Optional[Dict[str, float]] = None
    user_feedback: Optional[Dict[str, Any]] = None


class FeedbackCollector:
    """Collects and processes feedback for the PIB-VideoGen system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the FeedbackCollector.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.feedback_db_path = os.path.join('data', 'feedback')
        os.makedirs(self.feedback_db_path, exist_ok=True)
        
        # Initialize feedback storage
        self.current_month_file = self._get_current_month_file()
        
        logger.info(f"FeedbackCollector initialized with storage at {self.feedback_db_path}")
    
    def _get_current_month_file(self) -> str:
        """Get the filename for the current month's feedback data."""
        current_date = datetime.datetime.now()
        filename = f"feedback_{current_date.year}_{current_date.month:02d}.json"
        filepath = os.path.join(self.feedback_db_path, filename)
        
        # Create the file with empty array if it doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                json.dump([], f)
        
        return filepath
    
    def record_feedback(self, feedback: FeedbackMetrics) -> bool:
        """Record feedback for a generated video.
        
        Args:
            feedback: FeedbackMetrics object containing feedback data.
            
        Returns:
            bool: True if feedback was successfully recorded.
        """
        try:
            # Load existing feedback
            with open(self.current_month_file, 'r') as f:
                feedback_data = json.load(f)
            
            # Convert dataclass to dict
            feedback_dict = {
                'video_id': feedback.video_id,
                'timestamp': feedback.timestamp,
                'generation_time': feedback.generation_time,
                'visual_quality_score': feedback.visual_quality_score,
                'temporal_coherence_score': feedback.temporal_coherence_score,
                'factual_accuracy_score': feedback.factual_accuracy_score,
                'human_reviewer_id': feedback.human_reviewer_id,
                'human_review_notes': feedback.human_review_notes,
                'automated_metrics': feedback.automated_metrics,
                'user_feedback': feedback.user_feedback
            }
            
            # Add new feedback
            feedback_data.append(feedback_dict)
            
            # Save updated feedback
            with open(self.current_month_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            logger.info(f"Recorded feedback for video {feedback.video_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
    
    def get_recent_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent feedback entries.
        
        Args:
            limit: Maximum number of entries to return.
            
        Returns:
            List of feedback entries.
        """
        try:
            with open(self.current_month_file, 'r') as f:
                feedback_data = json.load(f)
            
            # Return most recent entries first
            return sorted(feedback_data, key=lambda x: x['timestamp'], reverse=True)[:limit]
        
        except Exception as e:
            logger.error(f"Error retrieving recent feedback: {e}")
            return []
    
    def calculate_metrics_summary(self) -> Dict[str, Any]:
        """Calculate summary metrics from feedback data.
        
        Returns:
            Dictionary containing summary metrics.
        """
        try:
            with open(self.current_month_file, 'r') as f:
                feedback_data = json.load(f)
            
            if not feedback_data:
                return {
                    'count': 0,
                    'message': 'No feedback data available'
                }
            
            # Extract metrics
            visual_scores = [entry['visual_quality_score'] for entry in feedback_data]
            temporal_scores = [entry['temporal_coherence_score'] for entry in feedback_data]
            factual_scores = [entry['factual_accuracy_score'] for entry in feedback_data]
            generation_times = [entry['generation_time'] for entry in feedback_data]
            
            # Calculate statistics
            summary = {
                'count': len(feedback_data),
                'visual_quality': {
                    'mean': np.mean(visual_scores),
                    'median': np.median(visual_scores),
                    'std': np.std(visual_scores)
                },
                'temporal_coherence': {
                    'mean': np.mean(temporal_scores),
                    'median': np.median(temporal_scores),
                    'std': np.std(temporal_scores)
                },
                'factual_accuracy': {
                    'mean': np.mean(factual_scores),
                    'median': np.median(factual_scores),
                    'std': np.std(factual_scores)
                },
                'generation_time': {
                    'mean': np.mean(generation_times),
                    'median': np.median(generation_times),
                    'std': np.std(generation_times),
                    'min': min(generation_times),
                    'max': max(generation_times)
                }
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error calculating metrics summary: {e}")
            return {
                'error': str(e),
                'message': 'Failed to calculate metrics summary'
            }