#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Improvement Pipeline Module

This module provides components for applying feedback insights to improve
the video generation models and system through automated and guided processes.
"""

import os
import json
import logging
import datetime
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field

from .feedback_analyzer import FeedbackInsight, FeedbackMetric

# Configure logging
logger = logging.getLogger(__name__)


class ImprovementStrategy(Enum):
    """Enumeration of model improvement strategies."""
    PARAMETER_TUNING = auto()      # Adjust model parameters
    FINE_TUNING = auto()           # Fine-tune model on specific data
    ARCHITECTURE_CHANGE = auto()   # Modify model architecture
    PIPELINE_OPTIMIZATION = auto() # Optimize processing pipeline
    DATA_AUGMENTATION = auto()     # Add more training data
    ENSEMBLE_METHODS = auto()      # Combine multiple models
    DISTILLATION = auto()          # Create smaller, faster models
    QUANTIZATION = auto()          # Reduce model precision for speed


@dataclass
class ImprovementAction:
    """Represents a specific action to improve the model or system."""
    id: str
    strategy: ImprovementStrategy
    description: str
    source_insights: List[str]  # IDs of insights that led to this action
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: Optional[datetime.datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    assigned_to: Optional[str] = None
    priority: int = 0  # Higher number means higher priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert the action to a dictionary."""
        return {
            "id": self.id,
            "strategy": self.strategy.name,
            "description": self.description,
            "source_insights": self.source_insights,
            "parameters": self.parameters,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "results": self.results,
            "assigned_to": self.assigned_to,
            "priority": self.priority
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImprovementAction':
        """Create an action from a dictionary."""
        return cls(
            id=data["id"],
            strategy=ImprovementStrategy[data["strategy"]],
            description=data["description"],
            source_insights=data["source_insights"],
            parameters=data.get("parameters", {}),
            status=data.get("status", "pending"),
            created_at=datetime.datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            results=data.get("results", {}),
            assigned_to=data.get("assigned_to"),
            priority=data.get("priority", 0)
        )


class ModelImprovementPipeline:
    """Manages the process of applying insights to improve models and systems."""

    def __init__(self, storage_dir: str = "feedback/improvements"):
        """Initialize the model improvement pipeline.

        Args:
            storage_dir: Directory to store improvement actions
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.actions: List[ImprovementAction] = []
        self.strategy_handlers: Dict[ImprovementStrategy, Callable] = {}
        self._load_existing_actions()
        logger.info(f"Initialized ModelImprovementPipeline with storage at {storage_dir}")

    def register_strategy_handler(self, 
                                 strategy: ImprovementStrategy, 
                                 handler: Callable[[ImprovementAction], Dict[str, Any]]) -> None:
        """Register a handler function for a specific improvement strategy.

        Args:
            strategy: The improvement strategy
            handler: Function that implements the strategy
        """
        self.strategy_handlers[strategy] = handler
        logger.info(f"Registered handler for strategy: {strategy.name}")

    def create_actions_from_insights(self, insights: List[FeedbackInsight]) -> List[ImprovementAction]:
        """Create improvement actions based on feedback insights.

        Args:
            insights: List of insights to process

        Returns:
            List of created improvement actions
        """
        new_actions = []

        for insight in insights:
            # Skip low-confidence insights
            if insight.confidence < 0.6:
                continue

            # Determine appropriate strategies based on the insight
            strategies = self._determine_strategies(insight)

            for strategy, params in strategies:
                # Generate a unique ID
                action_id = f"action_{strategy.name.lower()}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

                # Create the action
                action = ImprovementAction(
                    id=action_id,
                    strategy=strategy,
                    description=self._generate_action_description(insight, strategy),
                    source_insights=[insight.id],
                    parameters=params,
                    priority=insight.priority  # Inherit priority from insight
                )

                # Add to the list
                self.actions.append(action)
                new_actions.append(action)

                logger.info(f"Created improvement action {action_id} with strategy {strategy.name}")

        # Save actions
        if new_actions:
            self._save_actions()

        return new_actions

    def execute_action(self, action_id: str) -> Dict[str, Any]:
        """Execute an improvement action.

        Args:
            action_id: ID of the action to execute

        Returns:
            Results of the execution
        """
        # Find the action
        action = None
        for a in self.actions:
            if a.id == action_id:
                action = a
                break

        if not action:
            logger.error(f"Action {action_id} not found")
            return {"error": "Action not found"}

        # Check if there's a handler for this strategy
        if action.strategy not in self.strategy_handlers:
            logger.error(f"No handler registered for strategy {action.strategy.name}")
            return {"error": f"No handler for strategy {action.strategy.name}"}

        # Update status
        action.status = "in_progress"
        action.updated_at = datetime.datetime.now()
        self._save_actions()

        try:
            # Execute the handler
            handler = self.strategy_handlers[action.strategy]
            results = handler(action)

            # Update action with results
            action.results = results
            action.status = "completed"
            action.updated_at = datetime.datetime.now()
            self._save_actions()

            logger.info(f"Successfully executed action {action_id}")
            return results
        except Exception as e:
            # Handle failure
            action.status = "failed"
            action.results = {"error": str(e)}
            action.updated_at = datetime.datetime.now()
            self._save_actions()

            logger.error(f"Failed to execute action {action_id}: {e}")
            return {"error": str(e)}

    def get_actions(self, 
                   status: Optional[str] = None,
                   strategy: Optional[ImprovementStrategy] = None,
                   min_priority: int = 0) -> List[ImprovementAction]:
        """Get improvement actions with optional filtering.

        Args:
            status: Optional status filter
            strategy: Optional strategy filter
            min_priority: Minimum priority level

        Returns:
            List of actions matching the criteria
        """
        filtered_actions = []

        for action in self.actions:
            # Apply filters
            if status and action.status != status:
                continue
            if strategy and action.strategy != strategy:
                continue
            if action.priority < min_priority:
                continue

            filtered_actions.append(action)

        # Sort by priority (descending) and then by creation time (descending)
        filtered_actions.sort(key=lambda x: (-x.priority, -x.created_at.timestamp()))

        return filtered_actions

    def update_action_status(self, 
                           action_id: str, 
                           status: str,
                           results: Optional[Dict[str, Any]] = None) -> bool:
        """Update the status of an improvement action.

        Args:
            action_id: ID of the action
            status: New status
            results: Optional results to update

        Returns:
            True if successful, False otherwise
        """
        for action in self.actions:
            if action.id == action_id:
                action.status = status
                action.updated_at = datetime.datetime.now()
                
                if results:
                    action.results = results
                
                self._save_actions()
                logger.info(f"Updated action {action_id} status to {status}")
                return True

        logger.warning(f"Action {action_id} not found")
        return False

    def assign_action(self, action_id: str, assignee: str) -> bool:
        """Assign an improvement action to someone.

        Args:
            action_id: ID of the action
            assignee: Name or ID of the assignee

        Returns:
            True if successful, False otherwise
        """
        for action in self.actions:
            if action.id == action_id:
                action.assigned_to = assignee
                action.updated_at = datetime.datetime.now()
                self._save_actions()
                logger.info(f"Assigned action {action_id} to {assignee}")
                return True

        logger.warning(f"Action {action_id} not found")
        return False

    def _determine_strategies(self, insight: FeedbackInsight) -> List[Tuple[ImprovementStrategy, Dict[str, Any]]]:
        """Determine appropriate improvement strategies based on an insight.

        Args:
            insight: The insight to analyze

        Returns:
            List of tuples containing strategy and parameters
        """
        strategies = []

        # Map metrics to strategies
        if insight.metric == FeedbackMetric.GENERATION_TIME:
            # For slow generation, suggest optimization strategies
            if insight.value > 5.0:  # Assuming value is time in seconds
                strategies.append((ImprovementStrategy.PIPELINE_OPTIMIZATION, {
                    "target_time": max(1.0, insight.value * 0.7),  # Target 30% improvement
                    "focus_areas": ["preprocessing", "inference"]
                }))
                
                strategies.append((ImprovementStrategy.QUANTIZATION, {
                    "precision": "fp16",
                    "target_speedup": 1.5  # Target 50% speedup
                }))

        elif insight.metric == FeedbackMetric.QUALITY_SCORE:
            # For quality issues, suggest model improvements
            strategies.append((ImprovementStrategy.FINE_TUNING, {
                "target_area": "quality",
                "epochs": 5,
                "learning_rate": 1e-5
            }))
            
            if "temporal_inconsistency" in insight.tags:
                strategies.append((ImprovementStrategy.ARCHITECTURE_CHANGE, {
                    "component": "temporal_consistency",
                    "change_type": "enhance"
                }))

        elif insight.metric == FeedbackMetric.ERROR_RATE:
            # For high error rates, suggest robustness improvements
            strategies.append((ImprovementStrategy.DATA_AUGMENTATION, {
                "target_area": "robustness",
                "sample_count": 1000
            }))

        elif insight.metric == FeedbackMetric.RESOURCE_EFFICIENCY:
            # For high resource usage, suggest optimization
            strategies.append((ImprovementStrategy.DISTILLATION, {
                "teacher_model": "current",
                "target_size_reduction": 0.5  # Target 50% size reduction
            }))

        elif insight.metric == FeedbackMetric.EMERGENCY_PERFORMANCE:
            # For emergency mode performance issues
            strategies.append((ImprovementStrategy.PIPELINE_OPTIMIZATION, {
                "target_time": max(0.5, insight.value * 0.5),  # Target 50% improvement for emergency
                "focus_areas": ["emergency_pipeline"],
                "emergency_mode": True
            }))

        # If no specific strategies were determined, use parameter tuning as a fallback
        if not strategies:
            strategies.append((ImprovementStrategy.PARAMETER_TUNING, {
                "target_metric": insight.metric.name,
                "exploration_range": 0.2  # 20% parameter exploration
            }))

        return strategies

    def _generate_action_description(self, insight: FeedbackInsight, strategy: ImprovementStrategy) -> str:
        """Generate a description for an improvement action.

        Args:
            insight: The source insight
            strategy: The improvement strategy

        Returns:
            Description string
        """
        # Base description from insight
        base = f"Based on insight: {insight.description}"

        # Add strategy-specific details
        if strategy == ImprovementStrategy.PARAMETER_TUNING:
            return f"{base} - Tune model parameters to improve {insight.metric.name}"
        
        elif strategy == ImprovementStrategy.FINE_TUNING:
            return f"{base} - Fine-tune model on specific data to address quality issues"
        
        elif strategy == ImprovementStrategy.ARCHITECTURE_CHANGE:
            return f"{base} - Modify model architecture to improve performance or capabilities"
        
        elif strategy == ImprovementStrategy.PIPELINE_OPTIMIZATION:
            return f"{base} - Optimize processing pipeline for better efficiency"
        
        elif strategy == ImprovementStrategy.DATA_AUGMENTATION:
            return f"{base} - Add more training data to improve robustness"
        
        elif strategy == ImprovementStrategy.ENSEMBLE_METHODS:
            return f"{base} - Implement ensemble methods to improve overall quality"
        
        elif strategy == ImprovementStrategy.DISTILLATION:
            return f"{base} - Create smaller, faster models through distillation"
        
        elif strategy == ImprovementStrategy.QUANTIZATION:
            return f"{base} - Apply quantization to reduce model size and increase speed"
        
        else:
            return f"{base} - Apply {strategy.name} to address the identified issues"

    def _load_existing_actions(self) -> None:
        """Load existing improvement actions from storage."""
        self.actions = []

        # List all action files
        for filename in os.listdir(self.storage_dir):
            if not filename.startswith("actions_") or not filename.endswith(".json"):
                continue

            file_path = os.path.join(self.storage_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    actions_dict = json.load(f)

                for action_dict in actions_dict:
                    try:
                        action = ImprovementAction.from_dict(action_dict)
                        self.actions.append(action)
                    except Exception as e:
                        logger.error(f"Error parsing action: {e}")
            except Exception as e:
                logger.error(f"Error loading actions file {file_path}: {e}")

        logger.info(f"Loaded {len(self.actions)} existing improvement actions")

    def _save_actions(self) -> None:
        """Save current improvement actions to storage."""
        if not self.actions:
            return

        # Create a timestamp-based filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.storage_dir, f"actions_{timestamp}.json")

        # Convert actions to dictionaries
        actions_dict = [action.to_dict() for action in self.actions]

        # Save to file
        with open(filename, 'w') as f:
            json.dump(actions_dict, f, indent=2)

        logger.info(f"Saved {len(self.actions)} improvement actions to {filename}")

        # Clean up old files (keep only the 5 most recent)
        self._cleanup_old_files()

    def _cleanup_old_files(self, keep_count: int = 5) -> None:
        """Clean up old action files, keeping only the most recent ones.

        Args:
            keep_count: Number of recent files to keep
        """
        # List all action files
        action_files = []
        for filename in os.listdir(self.storage_dir):
            if filename.startswith("actions_") and filename.endswith(".json"):
                file_path = os.path.join(self.storage_dir, filename)
                action_files.append((file_path, os.path.getmtime(file_path)))

        # Sort by modification time (newest first)
        action_files.sort(key=lambda x: x[1], reverse=True)

        # Remove old files
        for file_path, _ in action_files[keep_count:]:
            try:
                os.remove(file_path)
                logger.debug(f"Removed old actions file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {e}")