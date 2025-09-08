#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feedback Collector Module

This module provides components for collecting feedback from various sources
to continuously improve the video generation system.
"""

import os
import json
import logging
import datetime
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)


class FeedbackSource(Enum):
    """Enumeration of feedback sources."""
    USER = auto()           # Direct user feedback
    SYSTEM = auto()         # System-generated feedback
    QUALITY_CHECK = auto()  # Automated quality assessment
    A_B_TEST = auto()       # A/B testing results
    PRODUCTION = auto()     # Production metrics
    EMERGENCY = auto()      # Emergency mode feedback


class FeedbackType(Enum):
    """Enumeration of feedback types."""
    QUALITY = auto()        # Video quality feedback
    PERFORMANCE = auto()    # Performance metrics
    ACCURACY = auto()       # Content accuracy
    USABILITY = auto()      # User experience
    TECHNICAL = auto()      # Technical issues
    CREATIVE = auto()       # Creative suggestions
    COMPLIANCE = auto()     # Compliance with guidelines


@dataclass
class FeedbackItem:
    """Represents a single feedback item."""
    id: str
    source: FeedbackSource
    type: FeedbackType
    timestamp: datetime.datetime
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    rating: Optional[int] = None  # Optional numerical rating (e.g., 1-5)
    tags: List[str] = field(default_factory=list)
    related_video_id: Optional[str] = None
    related_request_id: Optional[str] = None
    processed: bool = False
    priority: int = 0  # Higher number means higher priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert the feedback item to a dictionary."""
        return {
            "id": self.id,
            "source": self.source.name,
            "type": self.type.name,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "metadata": self.metadata,
            "rating": self.rating,
            "tags": self.tags,
            "related_video_id": self.related_video_id,
            "related_request_id": self.related_request_id,
            "processed": self.processed,
            "priority": self.priority
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackItem':
        """Create a feedback item from a dictionary."""
        return cls(
            id=data["id"],
            source=FeedbackSource[data["source"]],
            type=FeedbackType[data["type"]],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            rating=data.get("rating"),
            tags=data.get("tags", []),
            related_video_id=data.get("related_video_id"),
            related_request_id=data.get("related_request_id"),
            processed=data.get("processed", False),
            priority=data.get("priority", 0)
        )


class FeedbackCollector:
    """Collects and stores feedback from various sources."""

    def __init__(self, storage_dir: str = "feedback/data"):
        """Initialize the feedback collector.

        Args:
            storage_dir: Directory to store feedback data
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.current_batch: List[FeedbackItem] = []
        self.batch_size = 50  # Number of items to collect before saving
        logger.info(f"Initialized FeedbackCollector with storage at {storage_dir}")

    def collect(self, 
                source: Union[FeedbackSource, str],
                feedback_type: Union[FeedbackType, str],
                content: Dict[str, Any],
                rating: Optional[int] = None,
                tags: Optional[List[str]] = None,
                related_video_id: Optional[str] = None,
                related_request_id: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None,
                priority: int = 0) -> str:
        """Collect a feedback item.

        Args:
            source: Source of the feedback
            feedback_type: Type of feedback
            content: Feedback content
            rating: Optional numerical rating
            tags: Optional list of tags
            related_video_id: ID of related video
            related_request_id: ID of related request
            metadata: Additional metadata
            priority: Priority level (higher = more important)

        Returns:
            ID of the created feedback item
        """
        # Convert string enums to enum objects if needed
        if isinstance(source, str):
            source = FeedbackSource[source]
        if isinstance(feedback_type, str):
            feedback_type = FeedbackType[feedback_type]

        # Generate a unique ID
        feedback_id = f"fb_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.current_batch)}"

        # Create the feedback item
        feedback_item = FeedbackItem(
            id=feedback_id,
            source=source,
            type=feedback_type,
            timestamp=datetime.datetime.now(),
            content=content,
            rating=rating,
            tags=tags or [],
            related_video_id=related_video_id,
            related_request_id=related_request_id,
            metadata=metadata or {},
            priority=priority
        )

        # Add to the current batch
        self.current_batch.append(feedback_item)
        logger.debug(f"Collected feedback item {feedback_id} from {source.name}")

        # Save batch if it reaches the threshold
        if len(self.current_batch) >= self.batch_size:
            self.save_batch()

        return feedback_id

    def collect_user_feedback(self, 
                             user_id: str, 
                             video_id: str, 
                             rating: int,
                             comments: str,
                             tags: Optional[List[str]] = None) -> str:
        """Collect feedback directly from a user.

        Args:
            user_id: ID of the user providing feedback
            video_id: ID of the video being rated
            rating: Numerical rating (typically 1-5)
            comments: User comments
            tags: Optional list of tags

        Returns:
            ID of the created feedback item
        """
        content = {
            "user_id": user_id,
            "comments": comments
        }
        metadata = {"collection_method": "direct_user_input"}
        
        return self.collect(
            source=FeedbackSource.USER,
            feedback_type=FeedbackType.QUALITY,
            content=content,
            rating=rating,
            tags=tags,
            related_video_id=video_id,
            metadata=metadata,
            priority=3  # User feedback is high priority
        )

    def collect_system_metrics(self,
                              request_id: str,
                              metrics: Dict[str, Any],
                              video_id: Optional[str] = None) -> str:
        """Collect system performance metrics.

        Args:
            request_id: ID of the generation request
            metrics: Dictionary of performance metrics
            video_id: Optional ID of the generated video

        Returns:
            ID of the created feedback item
        """
        return self.collect(
            source=FeedbackSource.SYSTEM,
            feedback_type=FeedbackType.PERFORMANCE,
            content=metrics,
            related_request_id=request_id,
            related_video_id=video_id,
            metadata={"collection_method": "automated_metrics"}
        )

    def collect_quality_assessment(self,
                                  video_id: str,
                                  assessment_results: Dict[str, Any],
                                  request_id: Optional[str] = None) -> str:
        """Collect automated quality assessment results.

        Args:
            video_id: ID of the assessed video
            assessment_results: Results of quality assessment
            request_id: Optional ID of the generation request

        Returns:
            ID of the created feedback item
        """
        return self.collect(
            source=FeedbackSource.QUALITY_CHECK,
            feedback_type=FeedbackType.QUALITY,
            content=assessment_results,
            related_video_id=video_id,
            related_request_id=request_id,
            metadata={"collection_method": "automated_assessment"}
        )

    def collect_emergency_feedback(self,
                                  emergency_type: str,
                                  performance_metrics: Dict[str, Any],
                                  video_id: Optional[str] = None,
                                  request_id: Optional[str] = None) -> str:
        """Collect feedback specific to emergency mode operations.

        Args:
            emergency_type: Type of emergency
            performance_metrics: Performance metrics in emergency mode
            video_id: Optional ID of the generated video
            request_id: Optional ID of the generation request

        Returns:
            ID of the created feedback item
        """
        content = {
            "emergency_type": emergency_type,
            **performance_metrics
        }
        
        return self.collect(
            source=FeedbackSource.EMERGENCY,
            feedback_type=FeedbackType.PERFORMANCE,
            content=content,
            related_video_id=video_id,
            related_request_id=request_id,
            metadata={"collection_method": "emergency_mode_metrics"},
            priority=4  # Emergency feedback is high priority
        )

    def save_batch(self) -> None:
        """Save the current batch of feedback items to storage."""
        if not self.current_batch:
            return

        # Create a timestamp-based filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.storage_dir, f"feedback_batch_{timestamp}.json")

        # Convert feedback items to dictionaries
        items_dict = [item.to_dict() for item in self.current_batch]

        # Save to file
        with open(filename, 'w') as f:
            json.dump(items_dict, f, indent=2)

        logger.info(f"Saved {len(self.current_batch)} feedback items to {filename}")
        self.current_batch = []

    def load_feedback(self, 
                     start_date: Optional[datetime.datetime] = None,
                     end_date: Optional[datetime.datetime] = None,
                     source: Optional[FeedbackSource] = None,
                     feedback_type: Optional[FeedbackType] = None) -> List[FeedbackItem]:
        """Load feedback items from storage with optional filtering.

        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            source: Optional source filter
            feedback_type: Optional type filter

        Returns:
            List of feedback items matching the criteria
        """
        # Save any pending items first
        if self.current_batch:
            self.save_batch()

        all_items = []

        # List all feedback files
        for filename in os.listdir(self.storage_dir):
            if not filename.startswith("feedback_batch_") or not filename.endswith(".json"):
                continue

            file_path = os.path.join(self.storage_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    items_dict = json.load(f)
                    
                for item_dict in items_dict:
                    try:
                        item = FeedbackItem.from_dict(item_dict)
                        
                        # Apply filters
                        if start_date and item.timestamp < start_date:
                            continue
                        if end_date and item.timestamp > end_date:
                            continue
                        if source and item.source != source:
                            continue
                        if feedback_type and item.type != feedback_type:
                            continue
                            
                        all_items.append(item)
                    except Exception as e:
                        logger.error(f"Error parsing feedback item: {e}")
            except Exception as e:
                logger.error(f"Error loading feedback file {file_path}: {e}")

        logger.info(f"Loaded {len(all_items)} feedback items matching criteria")
        return all_items

    def mark_as_processed(self, feedback_id: str) -> bool:
        """Mark a feedback item as processed.

        Args:
            feedback_id: ID of the feedback item

        Returns:
            True if successful, False otherwise
        """
        # Check current batch first
        for item in self.current_batch:
            if item.id == feedback_id:
                item.processed = True
                logger.debug(f"Marked feedback item {feedback_id} as processed (in current batch)")
                return True

        # If not in current batch, need to search in stored files
        for filename in os.listdir(self.storage_dir):
            if not filename.startswith("feedback_batch_") or not filename.endswith(".json"):
                continue

            file_path = os.path.join(self.storage_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    items_dict = json.load(f)
                
                updated = False
                for item_dict in items_dict:
                    if item_dict.get("id") == feedback_id:
                        item_dict["processed"] = True
                        updated = True
                        break
                
                if updated:
                    with open(file_path, 'w') as f:
                        json.dump(items_dict, f, indent=2)
                    logger.debug(f"Marked feedback item {feedback_id} as processed (in storage)")
                    return True
            except Exception as e:
                logger.error(f"Error updating feedback file {file_path}: {e}")

        logger.warning(f"Feedback item {feedback_id} not found")
        return False