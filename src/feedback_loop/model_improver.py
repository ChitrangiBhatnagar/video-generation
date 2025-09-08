import os
import logging
import datetime
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelImprover:
    """Uses feedback data to improve video generation models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the ModelImprover.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.feedback_db_path = os.path.join('data', 'feedback')
        self.model_path = os.path.join('models', 'fine_tuned')
        self.improvement_logs_path = os.path.join('data', 'improvement_logs')
        
        # Create necessary directories
        os.makedirs(self.improvement_logs_path, exist_ok=True)
        
        # Set improvement thresholds
        self.min_feedback_entries = 50  # Minimum number of feedback entries needed before improvement
        self.improvement_interval_days = 7  # How often to check for improvements
        self.last_improvement_check = self._load_last_improvement_timestamp()
        
        logger.info(f"ModelImprover initialized")
    
    def _load_last_improvement_timestamp(self) -> datetime.datetime:
        """Load the timestamp of the last improvement check."""
        timestamp_file = os.path.join(self.improvement_logs_path, 'last_improvement.json')
        
        if os.path.exists(timestamp_file):
            try:
                with open(timestamp_file, 'r') as f:
                    data = json.load(f)
                return datetime.datetime.fromisoformat(data['timestamp'])
            except Exception as e:
                logger.error(f"Error loading last improvement timestamp: {e}")
        
        # Default to a week ago if no timestamp exists
        return datetime.datetime.now() - datetime.timedelta(days=self.improvement_interval_days)
    
    def _save_improvement_timestamp(self):
        """Save the current timestamp as the last improvement check."""
        timestamp_file = os.path.join(self.improvement_logs_path, 'last_improvement.json')
        
        try:
            with open(timestamp_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.error(f"Error saving improvement timestamp: {e}")
    
    def _collect_feedback_data(self) -> List[Dict[str, Any]]:
        """Collect all feedback data from the past month."""
        all_feedback = []
        
        # Get current month file
        current_date = datetime.datetime.now()
        current_month_file = f"feedback_{current_date.year}_{current_date.month:02d}.json"
        current_filepath = os.path.join(self.feedback_db_path, current_month_file)
        
        # Get previous month file
        prev_date = current_date - datetime.timedelta(days=30)
        prev_month_file = f"feedback_{prev_date.year}_{prev_date.month:02d}.json"
        prev_filepath = os.path.join(self.feedback_db_path, prev_month_file)
        
        # Load current month feedback
        if os.path.exists(current_filepath):
            try:
                with open(current_filepath, 'r') as f:
                    all_feedback.extend(json.load(f))
            except Exception as e:
                logger.error(f"Error loading current month feedback: {e}")
        
        # Load previous month feedback
        if os.path.exists(prev_filepath):
            try:
                with open(prev_filepath, 'r') as f:
                    all_feedback.extend(json.load(f))
            except Exception as e:
                logger.error(f"Error loading previous month feedback: {e}")
        
        return all_feedback
    
    def _analyze_feedback_for_improvements(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feedback data to identify potential model improvements.
        
        Args:
            feedback_data: List of feedback entries.
            
        Returns:
            Dictionary containing improvement recommendations.
        """
        if len(feedback_data) < self.min_feedback_entries:
            return {
                'status': 'insufficient_data',
                'message': f"Need at least {self.min_feedback_entries} feedback entries (have {len(feedback_data)})"
            }
        
        # Extract scores
        visual_scores = [entry['visual_quality_score'] for entry in feedback_data]
        temporal_scores = [entry['temporal_coherence_score'] for entry in feedback_data]
        factual_scores = [entry['factual_accuracy_score'] for entry in feedback_data]
        
        # Calculate mean scores
        mean_visual = np.mean(visual_scores)
        mean_temporal = np.mean(temporal_scores)
        mean_factual = np.mean(factual_scores)
        
        # Identify areas for improvement
        improvements = []
        
        if mean_visual < 7.0:  # Below 7.0 on a 10-point scale
            improvements.append({
                'component': 'visual_quality',
                'current_score': mean_visual,
                'target_score': 8.0,
                'recommendation': 'Increase base_noise_lambda in VideoFusion model',
                'parameter_adjustments': {
                    'base_noise_lambda': min(0.8, 0.7 + (7.0 - mean_visual) * 0.05)
                }
            })
        
        if mean_temporal < 7.0:
            improvements.append({
                'component': 'temporal_coherence',
                'current_score': mean_temporal,
                'target_score': 8.0,
                'recommendation': 'Adjust residual_noise_lambda in VideoFusion model',
                'parameter_adjustments': {
                    'residual_noise_lambda': min(0.4, 0.3 + (7.0 - mean_temporal) * 0.05)
                }
            })
        
        if mean_factual < 8.0:  # Higher standard for factual accuracy
            improvements.append({
                'component': 'factual_accuracy',
                'current_score': mean_factual,
                'target_score': 9.0,
                'recommendation': 'Enhance fact checking module confidence threshold',
                'parameter_adjustments': {
                    'fact_check_confidence_threshold': min(0.9, 0.7 + (8.0 - mean_factual) * 0.1)
                }
            })
        
        return {
            'status': 'analysis_complete',
            'feedback_count': len(feedback_data),
            'mean_scores': {
                'visual_quality': mean_visual,
                'temporal_coherence': mean_temporal,
                'factual_accuracy': mean_factual
            },
            'improvements': improvements
        }
    
    def _update_model_parameters(self, improvements: List[Dict[str, Any]]) -> bool:
        """Update model parameters based on improvement recommendations.
        
        Args:
            improvements: List of improvement recommendations.
            
        Returns:
            bool: True if parameters were successfully updated.
        """
        if not improvements:
            logger.info("No improvements needed")
            return False
        
        try:
            # Load current config
            config_path = os.path.join('config', 'default_config.yaml')
            with open(config_path, 'r') as f:
                import yaml
                config = yaml.safe_load(f)
            
            # Apply parameter adjustments
            parameter_changes = {}
            for improvement in improvements:
                for param, value in improvement['parameter_adjustments'].items():
                    if param in config:
                        old_value = config[param]
                        config[param] = value
                        parameter_changes[param] = {'old': old_value, 'new': value}
                        logger.info(f"Updated {param}: {old_value} -> {value}")
            
            # Save improvement history and config
            improvement_details = {
                'parameter_changes': parameter_changes,
                'description': 'Model parameters updated based on feedback analysis'
            }
            
            self._save_improvement_history(improvement_details, config)
            
            logger.info(f"Model parameters updated and saved to improvement logs")
            return True
        
        except Exception as e:
            logger.error(f"Error updating model parameters: {e}")
            return False
    
    def check_for_improvements(self) -> Dict[str, Any]:
        """Check if model improvements are needed based on feedback data.
        
        Returns:
            Dictionary containing improvement status and details.
        """
        # Check if it's time for an improvement check
        now = datetime.datetime.now()
        days_since_last_check = (now - self.last_improvement_check).days
        
        if days_since_last_check < self.improvement_interval_days:
            return {
                'status': 'too_soon',
                'message': f"Next check scheduled in {self.improvement_interval_days - days_since_last_check} days"
            }
        
        # Collect feedback data
        feedback_data = self._collect_feedback_data()
        
        # Analyze feedback for improvements
        analysis = self._analyze_feedback_for_improvements(feedback_data)
        
        # Update model parameters if improvements are recommended
        if analysis['status'] == 'analysis_complete' and analysis['improvements']:
            # Load current config
            config_path = os.path.join('config', 'default_config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            updated = self._update_model_parameters(analysis['improvements'])
            if updated:
                # Save the improved configuration
                improvement_details = {
                    'parameter_changes': analysis.get('improvements', []),
                    'description': 'Model parameters updated based on feedback analysis'
                }
                self._save_improvement_history(improvement_details, config)
                analysis['status'] = 'improvements_applied'
        
        # Update last improvement check timestamp
        self._save_improvement_timestamp()
        self.last_improvement_check = now
        
        return analysis
    
    def trigger_immediate_improvement_check(self) -> Dict[str, Any]:
        """Trigger an immediate improvement check, bypassing the interval check.
        
        Returns:
            Dictionary containing improvement status and details.
        """
        # Force the last check timestamp to be old enough
        self.last_improvement_check = datetime.datetime.now() - datetime.timedelta(days=self.improvement_interval_days + 1)
        
        # Run the improvement check
        return self.check_for_improvements()
        
    def _save_improvement_history(self, improvements: Dict[str, Any], config: Dict[str, Any]):
        """Save the improvement history and improved configuration to files.
        
        Args:
            improvements: Dictionary containing improvement details.
            config: The improved configuration.
        """
        try:
            # Create timestamp for this improvement
            timestamp = datetime.datetime.now().isoformat()
            timestamp_safe = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create directory for improvement logs if it doesn't exist
            os.makedirs(self.improvement_logs_path, exist_ok=True)
            
            # Save the improved configuration
            config_filename = f"improved_config_{timestamp_safe}.yaml"
            config_path = os.path.join(self.improvement_logs_path, config_filename)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Create the improvement log entry
            entry = {
                'timestamp': timestamp,
                'parameter_changes': improvements.get('parameter_changes', {}),
                'description': improvements.get('description', 'Model parameters updated based on feedback'),
                'config_file': config_path
            }
            
            # Save the improvement log
            log_filename = f"improvement_log_{timestamp_safe}.json"
            log_path = os.path.join(self.improvement_logs_path, log_filename)
            
            with open(log_path, 'w') as f:
                json.dump(entry, f, indent=2)
                
            logger.info(f"Saved improvement history entry: {entry['description']}")
            logger.info(f"Saved improved configuration to: {config_path}")
            
            return entry
        except Exception as e:
            logger.error(f"Error saving improvement history: {e}")
            return None
            
    def get_improvement_history(self) -> List[Dict[str, Any]]:
        """Get the history of model improvements.
        
        Returns:
            List of improvement entries.
        """
        try:
            history_files = [f for f in os.listdir(self.improvement_logs_path) 
                           if f.startswith('improvement_log_') and f.endswith('.json')]
            
            history = []
            for file in history_files:
                try:
                    with open(os.path.join(self.improvement_logs_path, file), 'r') as f:
                        history.append(json.load(f))
                except Exception as e:
                    logger.error(f"Error loading improvement history file {file}: {e}")
            
            return sorted(history, key=lambda x: x['timestamp'], reverse=True)
        except Exception as e:
            logger.error(f"Error loading improvement history: {e}")
            return []
            
    def get_latest_improved_config(self) -> Optional[Dict[str, Any]]:
        """Get the latest improved configuration if available.
        
        Returns:
            The latest improved configuration or None if no improvements have been made.
        """
        try:
            history = self.get_improvement_history()
            if not history:
                return None
                
            # Get the latest improvement entry
            latest_entry = history[0]
            config_file = latest_entry.get('config_file')
            
            if config_file and os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    import yaml
                    return yaml.safe_load(f)
            return None
        except Exception as e:
            logger.error(f"Error getting latest improved config: {e}")
            return None