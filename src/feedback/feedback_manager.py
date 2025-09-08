#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feedback Manager Module

This module provides the central coordination for the feedback loop system,
managing the collection, analysis, and application of feedback for continuous improvement.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

from .feedback_collector import FeedbackCollector, FeedbackSource, FeedbackType, FeedbackItem
from .feedback_analyzer import FeedbackAnalyzer, FeedbackInsight, FeedbackMetric

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FeedbackConfig:
    """Configuration for the feedback system."""
    storage_dir: str = "feedback/data"
    analysis_interval_hours: int = 24  # How often to run analysis
    auto_apply_insights: bool = False  # Whether to automatically apply insights
    min_confidence_threshold: float = 0.7  # Minimum confidence for auto-application
    retention_days: int = 90  # How long to keep feedback data
    collect_system_metrics: bool = True  # Whether to collect system metrics
    collect_user_feedback: bool = True  # Whether to collect user feedback
    collect_quality_metrics: bool = True  # Whether to collect quality metrics
    emergency_feedback_priority: int = 4  # Priority for emergency feedback
    export_insights: bool = True  # Whether to export insights to files
    insights_export_dir: str = "feedback/insights"  # Directory for exporting insights
    notification_channels: List[str] = field(default_factory=lambda: ["log"])  # Channels for notifications


class FeedbackManager:
    """Central manager for the feedback loop system."""

    def __init__(self, config: Optional[FeedbackConfig] = None):
        """Initialize the feedback manager.

        Args:
            config: Optional configuration for the feedback system
        """
        self.config = config or FeedbackConfig()
        self.collector = FeedbackCollector(storage_dir=self.config.storage_dir)
        self.analyzer = FeedbackAnalyzer()
        self.last_analysis_time: Optional[datetime.datetime] = None
        self.pending_insights: List[FeedbackInsight] = []
        self.applied_insights: List[Tuple[FeedbackInsight, datetime.datetime]] = []
        
        # Create export directory if needed
        if self.config.export_insights:
            os.makedirs(self.config.insights_export_dir, exist_ok=True)
        
        logger.info(f"Initialized FeedbackManager with storage at {self.config.storage_dir}")

    def collect_feedback(self, 
                        source: Union[FeedbackSource, str],
                        feedback_type: Union[FeedbackType, str],
                        content: Dict[str, Any],
                        **kwargs) -> str:
        """Collect feedback using the collector.

        Args:
            source: Source of the feedback
            feedback_type: Type of feedback
            content: Feedback content
            **kwargs: Additional arguments for the collector

        Returns:
            ID of the created feedback item
        """
        # Check if collection is enabled for this source
        if source == FeedbackSource.SYSTEM and not self.config.collect_system_metrics:
            logger.debug("System metrics collection is disabled")
            return ""
        
        if source == FeedbackSource.USER and not self.config.collect_user_feedback:
            logger.debug("User feedback collection is disabled")
            return ""
        
        if source == FeedbackSource.QUALITY_CHECK and not self.config.collect_quality_metrics:
            logger.debug("Quality metrics collection is disabled")
            return ""
        
        # Set priority for emergency feedback
        if source == FeedbackSource.EMERGENCY:
            kwargs["priority"] = self.config.emergency_feedback_priority
        
        # Collect the feedback
        feedback_id = self.collector.collect(source, feedback_type, content, **kwargs)
        logger.debug(f"Collected feedback with ID {feedback_id}")
        
        # Check if we should run analysis
        self._check_analysis_schedule()
        
        return feedback_id

    def run_analysis(self, force: bool = False) -> List[FeedbackInsight]:
        """Run analysis on collected feedback.

        Args:
            force: Whether to force analysis even if the interval hasn't elapsed

        Returns:
            List of new insights generated
        """
        current_time = datetime.datetime.now()
        
        # Check if analysis interval has elapsed
        if not force and self.last_analysis_time:
            elapsed_hours = (current_time - self.last_analysis_time).total_seconds() / 3600
            if elapsed_hours < self.config.analysis_interval_hours:
                logger.debug(f"Analysis interval not elapsed. Next analysis in {self.config.analysis_interval_hours - elapsed_hours:.1f} hours")
                return []
        
        # Load feedback since last analysis
        start_date = self.last_analysis_time if self.last_analysis_time else None
        feedback_items = self.collector.load_feedback(start_date=start_date)
        
        if not feedback_items:
            logger.info("No new feedback items to analyze")
            self.last_analysis_time = current_time
            return []
        
        # Run analysis
        new_insights = self.analyzer.analyze_batch(feedback_items)
        self.last_analysis_time = current_time
        
        # Add to pending insights
        self.pending_insights.extend(new_insights)
        
        # Export insights if configured
        if self.config.export_insights and new_insights:
            self._export_insights(new_insights)
        
        # Check for auto-application
        if self.config.auto_apply_insights:
            self._auto_apply_insights()
        
        # Send notifications
        self._send_insight_notifications(new_insights)
        
        logger.info(f"Analysis complete. Generated {len(new_insights)} new insights")
        return new_insights

    def get_pending_insights(self, 
                           min_confidence: Optional[float] = None,
                           min_priority: int = 0) -> List[FeedbackInsight]:
        """Get pending insights with optional filtering.

        Args:
            min_confidence: Optional minimum confidence threshold
            min_priority: Minimum priority level

        Returns:
            List of pending insights matching the criteria
        """
        filtered_insights = []
        
        for insight in self.pending_insights:
            if min_confidence is not None and insight.confidence < min_confidence:
                continue
            if insight.priority < min_priority:
                continue
            filtered_insights.append(insight)
        
        # Sort by priority (descending) and then by timestamp (descending)
        filtered_insights.sort(key=lambda x: (-x.priority, -x.timestamp.timestamp()))
        
        return filtered_insights

    def mark_insight_applied(self, insight_id: str, notes: Optional[str] = None) -> bool:
        """Mark an insight as applied.

        Args:
            insight_id: ID of the insight
            notes: Optional notes about the application

        Returns:
            True if successful, False otherwise
        """
        for i, insight in enumerate(self.pending_insights):
            if insight.id == insight_id:
                # Add notes to metadata if provided
                if notes:
                    insight.metadata["application_notes"] = notes
                
                # Move from pending to applied
                self.applied_insights.append((insight, datetime.datetime.now()))
                self.pending_insights.pop(i)
                
                logger.info(f"Marked insight {insight_id} as applied")
                return True
        
        logger.warning(f"Insight {insight_id} not found in pending insights")
        return False

    def get_applied_insights(self, 
                           start_date: Optional[datetime.datetime] = None,
                           end_date: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """Get insights that have been applied.

        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering

        Returns:
            List of applied insights with application timestamps
        """
        filtered_insights = []
        
        for insight, applied_time in self.applied_insights:
            if start_date and applied_time < start_date:
                continue
            if end_date and applied_time > end_date:
                continue
            
            insight_dict = insight.to_dict()
            insight_dict["applied_time"] = applied_time.isoformat()
            filtered_insights.append(insight_dict)
        
        # Sort by application time (descending)
        filtered_insights.sort(key=lambda x: x["applied_time"], reverse=True)
        
        return filtered_insights

    def get_feedback_summary(self, 
                           days: int = 30,
                           include_insights: bool = True) -> Dict[str, Any]:
        """Get a summary of feedback and insights.

        Args:
            days: Number of days to include in the summary
            include_insights: Whether to include insights in the summary

        Returns:
            Dictionary with feedback summary
        """
        start_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        # Load feedback
        feedback_items = self.collector.load_feedback(start_date=start_date)
        
        # Count by source and type
        source_counts = {}
        type_counts = {}
        rating_sum = 0
        rating_count = 0
        
        for item in feedback_items:
            # Count by source
            source_name = item.source.name
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
            
            # Count by type
            type_name = item.type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            # Track ratings
            if item.rating is not None:
                rating_sum += item.rating
                rating_count += 1
        
        # Calculate average rating
        avg_rating = rating_sum / rating_count if rating_count > 0 else None
        
        # Get top insights if requested
        top_insights = []
        if include_insights:
            # Combine pending and applied insights
            all_insights = self.pending_insights + [insight for insight, _ in self.applied_insights]
            
            # Filter by date and sort by priority
            recent_insights = [insight for insight in all_insights if insight.timestamp >= start_date]
            recent_insights.sort(key=lambda x: (-x.priority, -x.timestamp.timestamp()))
            
            # Take top 5
            top_insights = [insight.to_dict() for insight in recent_insights[:5]]
        
        return {
            "period_days": days,
            "start_date": start_date.isoformat(),
            "end_date": datetime.datetime.now().isoformat(),
            "total_feedback_count": len(feedback_items),
            "by_source": source_counts,
            "by_type": type_counts,
            "average_rating": avg_rating,
            "rating_count": rating_count,
            "top_insights": top_insights
        }

    def cleanup_old_data(self) -> int:
        """Clean up old feedback data based on retention policy.

        Returns:
            Number of files removed
        """
        if not self.config.retention_days:
            logger.info("Retention policy not set, skipping cleanup")
            return 0
        
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.config.retention_days)
        removed_count = 0
        
        # List all feedback files
        for filename in os.listdir(self.config.storage_dir):
            if not filename.startswith("feedback_batch_") or not filename.endswith(".json"):
                continue
            
            file_path = os.path.join(self.config.storage_dir, filename)
            file_date_str = filename.replace("feedback_batch_", "").split("_")[0]
            
            try:
                # Parse date from filename (format: YYYYMMDD)
                file_date = datetime.datetime.strptime(file_date_str, "%Y%m%d")
                
                # Check if file is older than retention period
                if file_date < cutoff_date:
                    os.remove(file_path)
                    removed_count += 1
                    logger.debug(f"Removed old feedback file: {filename}")
            except Exception as e:
                logger.error(f"Error parsing date from filename {filename}: {e}")
        
        logger.info(f"Cleanup complete. Removed {removed_count} old feedback files")
        return removed_count

    def _check_analysis_schedule(self) -> None:
        """Check if analysis should be run based on the schedule."""
        if not self.last_analysis_time:
            # First time, run analysis immediately
            self.run_analysis()
            return
        
        current_time = datetime.datetime.now()
        elapsed_hours = (current_time - self.last_analysis_time).total_seconds() / 3600
        
        if elapsed_hours >= self.config.analysis_interval_hours:
            self.run_analysis()

    def _auto_apply_insights(self) -> None:
        """Automatically apply high-confidence insights."""
        if not self.config.auto_apply_insights:
            return
        
        auto_applied = []
        remaining = []
        
        for insight in self.pending_insights:
            if insight.confidence >= self.config.min_confidence_threshold:
                # Auto-apply this insight
                self.applied_insights.append((insight, datetime.datetime.now()))
                auto_applied.append(insight)
                logger.info(f"Auto-applied insight {insight.id} with confidence {insight.confidence}")
            else:
                remaining.append(insight)
        
        # Update pending insights
        self.pending_insights = remaining
        
        if auto_applied:
            logger.info(f"Auto-applied {len(auto_applied)} insights")

    def _export_insights(self, insights: List[FeedbackInsight]) -> None:
        """Export insights to files.

        Args:
            insights: List of insights to export
        """
        if not self.config.export_insights or not insights:
            return
        
        # Create a timestamp-based filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.config.insights_export_dir, f"insights_{timestamp}.json")
        
        # Convert insights to dictionaries
        insights_dict = [insight.to_dict() for insight in insights]
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(insights_dict, f, indent=2)
        
        logger.info(f"Exported {len(insights)} insights to {filename}")

    def _send_insight_notifications(self, insights: List[FeedbackInsight]) -> None:
        """Send notifications about new insights.

        Args:
            insights: List of new insights
        """
        if not insights or not self.config.notification_channels:
            return
        
        # Filter to high-priority insights
        high_priority = [insight for insight in insights if insight.priority >= 3]
        
        if not high_priority:
            return
        
        # Log notifications
        if "log" in self.config.notification_channels:
            for insight in high_priority:
                logger.warning(f"High-priority insight: {insight.description} (ID: {insight.id})")
        
        # Other notification channels could be implemented here (email, Slack, etc.)