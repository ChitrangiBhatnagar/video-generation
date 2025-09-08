#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feedback Analyzer Module

This module provides components for analyzing feedback data to extract
actionable insights for improving the video generation system.
"""

import logging
import datetime
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

from .feedback_collector import FeedbackItem, FeedbackSource, FeedbackType

# Configure logging
logger = logging.getLogger(__name__)


class FeedbackMetric(Enum):
    """Enumeration of feedback metrics that can be calculated."""
    AVERAGE_RATING = auto()          # Average user rating
    GENERATION_TIME = auto()         # Average generation time
    ERROR_RATE = auto()              # Rate of errors
    QUALITY_SCORE = auto()           # Composite quality score
    USER_SATISFACTION = auto()       # User satisfaction index
    EMERGENCY_PERFORMANCE = auto()   # Performance in emergency mode
    RESOURCE_EFFICIENCY = auto()     # Resource usage efficiency


@dataclass
class FeedbackInsight:
    """Represents an actionable insight derived from feedback analysis."""
    id: str
    timestamp: datetime.datetime
    metric: FeedbackMetric
    value: float
    confidence: float  # 0.0 to 1.0
    source_feedback_ids: List[str]
    description: str
    suggested_actions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher number means higher priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert the insight to a dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "metric": self.metric.name,
            "value": self.value,
            "confidence": self.confidence,
            "source_feedback_ids": self.source_feedback_ids,
            "description": self.description,
            "suggested_actions": self.suggested_actions,
            "tags": self.tags,
            "metadata": self.metadata,
            "priority": self.priority
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackInsight':
        """Create an insight from a dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            metric=FeedbackMetric[data["metric"]],
            value=data["value"],
            confidence=data["confidence"],
            source_feedback_ids=data["source_feedback_ids"],
            description=data["description"],
            suggested_actions=data.get("suggested_actions", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            priority=data.get("priority", 0)
        )


class FeedbackAnalyzer:
    """Analyzes feedback data to extract actionable insights."""

    def __init__(self):
        """Initialize the feedback analyzer."""
        self.insights: List[FeedbackInsight] = []
        logger.info("Initialized FeedbackAnalyzer")

    def analyze_batch(self, feedback_items: List[FeedbackItem]) -> List[FeedbackInsight]:
        """Analyze a batch of feedback items to generate insights.

        Args:
            feedback_items: List of feedback items to analyze

        Returns:
            List of generated insights
        """
        if not feedback_items:
            logger.warning("No feedback items to analyze")
            return []

        batch_insights = []

        # Group feedback by type for specialized analysis
        by_type = defaultdict(list)
        for item in feedback_items:
            by_type[item.type].append(item)

        # Analyze user ratings if available
        if FeedbackType.QUALITY in by_type:
            rating_insights = self._analyze_user_ratings(by_type[FeedbackType.QUALITY])
            batch_insights.extend(rating_insights)

        # Analyze performance metrics if available
        if FeedbackType.PERFORMANCE in by_type:
            perf_insights = self._analyze_performance_metrics(by_type[FeedbackType.PERFORMANCE])
            batch_insights.extend(perf_insights)

        # Analyze emergency feedback if available
        emergency_items = [item for item in feedback_items if item.source == FeedbackSource.EMERGENCY]
        if emergency_items:
            emergency_insights = self._analyze_emergency_feedback(emergency_items)
            batch_insights.extend(emergency_insights)

        # Analyze content accuracy if available
        if FeedbackType.ACCURACY in by_type:
            accuracy_insights = self._analyze_content_accuracy(by_type[FeedbackType.ACCURACY])
            batch_insights.extend(accuracy_insights)

        # Add all insights to the internal list
        self.insights.extend(batch_insights)
        logger.info(f"Generated {len(batch_insights)} insights from {len(feedback_items)} feedback items")

        return batch_insights

    def _analyze_user_ratings(self, feedback_items: List[FeedbackItem]) -> List[FeedbackInsight]:
        """Analyze user ratings to generate insights.

        Args:
            feedback_items: List of quality feedback items

        Returns:
            List of generated insights
        """
        insights = []

        # Filter items with ratings
        rated_items = [item for item in feedback_items if item.rating is not None]
        if not rated_items:
            return insights

        # Calculate average rating
        ratings = [item.rating for item in rated_items]
        avg_rating = sum(ratings) / len(ratings)

        # Generate insight for average rating
        timestamp = datetime.datetime.now()
        insight_id = f"insight_rating_{timestamp.strftime('%Y%m%d%H%M%S')}"
        
        # Determine confidence based on sample size
        confidence = min(0.5 + (len(rated_items) / 100), 0.95)  # Max confidence of 0.95
        
        # Generate description and actions based on rating
        if avg_rating >= 4.0:
            description = f"High user satisfaction with an average rating of {avg_rating:.2f}/5"
            actions = ["Maintain current quality standards", "Identify specific features users appreciate"]
            priority = 1
        elif avg_rating >= 3.0:
            description = f"Moderate user satisfaction with an average rating of {avg_rating:.2f}/5"
            actions = ["Identify improvement areas from user comments", "Consider A/B testing for key features"]
            priority = 2
        else:
            description = f"Low user satisfaction with an average rating of {avg_rating:.2f}/5"
            actions = ["Urgently review user comments for pain points", "Consider rolling back recent changes", "Increase quality testing"]
            priority = 3

        insight = FeedbackInsight(
            id=insight_id,
            timestamp=timestamp,
            metric=FeedbackMetric.AVERAGE_RATING,
            value=avg_rating,
            confidence=confidence,
            source_feedback_ids=[item.id for item in rated_items],
            description=description,
            suggested_actions=actions,
            tags=["user_satisfaction", "ratings"],
            priority=priority
        )
        
        insights.append(insight)

        # Analyze rating distribution
        if len(rated_items) >= 10:  # Only if we have enough data
            rating_counts = Counter(ratings)
            total = len(ratings)
            
            # Check for bimodal distribution (might indicate polarizing features)
            if rating_counts.get(1, 0) + rating_counts.get(2, 0) >= total * 0.25 and \
               rating_counts.get(4, 0) + rating_counts.get(5, 0) >= total * 0.25:
                
                insight_id = f"insight_polarized_{timestamp.strftime('%Y%m%d%H%M%S')}"
                description = "Polarized user ratings detected - some users love the system while others rate it poorly"
                actions = [
                    "Segment users to understand different use cases",
                    "Identify features that might be divisive",
                    "Consider personalization options to satisfy different user groups"
                ]
                
                polarization_insight = FeedbackInsight(
                    id=insight_id,
                    timestamp=timestamp,
                    metric=FeedbackMetric.USER_SATISFACTION,
                    value=0.0,  # No specific value for this insight
                    confidence=0.7,
                    source_feedback_ids=[item.id for item in rated_items],
                    description=description,
                    suggested_actions=actions,
                    tags=["user_satisfaction", "polarization", "segmentation"],
                    priority=2
                )
                
                insights.append(polarization_insight)

        return insights

    def _analyze_performance_metrics(self, feedback_items: List[FeedbackItem]) -> List[FeedbackInsight]:
        """Analyze performance metrics to generate insights.

        Args:
            feedback_items: List of performance feedback items

        Returns:
            List of generated insights
        """
        insights = []
        timestamp = datetime.datetime.now()

        # Extract generation times if available
        generation_times = []
        for item in feedback_items:
            if "generation_time" in item.content:
                generation_times.append(item.content["generation_time"])

        if generation_times:
            avg_time = sum(generation_times) / len(generation_times)
            insight_id = f"insight_gentime_{timestamp.strftime('%Y%m%d%H%M%S')}"
            
            # Determine description and actions based on average time
            if avg_time > 10.0:  # Assuming 10 seconds is a threshold for concern
                description = f"High average generation time of {avg_time:.2f} seconds"
                actions = [
                    "Optimize model inference",
                    "Consider model quantization or distillation",
                    "Review preprocessing pipeline for bottlenecks"
                ]
                priority = 3
            else:
                description = f"Acceptable average generation time of {avg_time:.2f} seconds"
                actions = ["Continue monitoring for performance regressions"]
                priority = 1
            
            insight = FeedbackInsight(
                id=insight_id,
                timestamp=timestamp,
                metric=FeedbackMetric.GENERATION_TIME,
                value=avg_time,
                confidence=0.8,
                source_feedback_ids=[item.id for item in feedback_items if "generation_time" in item.content],
                description=description,
                suggested_actions=actions,
                tags=["performance", "generation_time"],
                priority=priority
            )
            
            insights.append(insight)

        # Extract error rates if available
        error_count = 0
        total_count = len(feedback_items)
        for item in feedback_items:
            if "error" in item.content and item.content["error"]:
                error_count += 1

        if total_count > 0:
            error_rate = error_count / total_count
            insight_id = f"insight_errors_{timestamp.strftime('%Y%m%d%H%M%S')}"
            
            # Determine description and actions based on error rate
            if error_rate > 0.05:  # More than 5% error rate is concerning
                description = f"High error rate of {error_rate:.2%}"
                actions = [
                    "Review error logs for common patterns",
                    "Implement additional error handling",
                    "Consider rolling back recent changes if errors are new"
                ]
                priority = 4
            else:
                description = f"Acceptable error rate of {error_rate:.2%}"
                actions = ["Continue monitoring for error patterns"]
                priority = 1
            
            insight = FeedbackInsight(
                id=insight_id,
                timestamp=timestamp,
                metric=FeedbackMetric.ERROR_RATE,
                value=error_rate,
                confidence=0.85,
                source_feedback_ids=[item.id for item in feedback_items],
                description=description,
                suggested_actions=actions,
                tags=["performance", "errors", "reliability"],
                priority=priority
            )
            
            insights.append(insight)

        # Extract resource usage if available
        resource_items = [item for item in feedback_items if "resource_usage" in item.content]
        if resource_items:
            # Calculate average memory and GPU usage
            memory_usages = [item.content["resource_usage"].get("memory_mb", 0) for item in resource_items 
                            if "memory_mb" in item.content.get("resource_usage", {})]
            gpu_usages = [item.content["resource_usage"].get("gpu_utilization", 0) for item in resource_items 
                         if "gpu_utilization" in item.content.get("resource_usage", {})]
            
            if memory_usages:
                avg_memory = sum(memory_usages) / len(memory_usages)
                insight_id = f"insight_memory_{timestamp.strftime('%Y%m%d%H%M%S')}"
                
                if avg_memory > 8000:  # Assuming 8GB is a threshold for concern
                    description = f"High average memory usage of {avg_memory:.0f} MB"
                    actions = [
                        "Implement memory optimization techniques",
                        "Consider model pruning or quantization",
                        "Review for memory leaks"
                    ]
                    priority = 3
                else:
                    description = f"Acceptable average memory usage of {avg_memory:.0f} MB"
                    actions = ["Continue monitoring resource usage"]
                    priority = 1
                
                insight = FeedbackInsight(
                    id=insight_id,
                    timestamp=timestamp,
                    metric=FeedbackMetric.RESOURCE_EFFICIENCY,
                    value=avg_memory,
                    confidence=0.8,
                    source_feedback_ids=[item.id for item in resource_items],
                    description=description,
                    suggested_actions=actions,
                    tags=["performance", "memory_usage", "resource_efficiency"],
                    priority=priority
                )
                
                insights.append(insight)

        return insights

    def _analyze_emergency_feedback(self, feedback_items: List[FeedbackItem]) -> List[FeedbackInsight]:
        """Analyze emergency mode feedback to generate insights.

        Args:
            feedback_items: List of emergency feedback items

        Returns:
            List of generated insights
        """
        insights = []
        timestamp = datetime.datetime.now()

        if not feedback_items:
            return insights

        # Group by emergency type
        by_emergency_type = defaultdict(list)
        for item in feedback_items:
            if "emergency_type" in item.content:
                by_emergency_type[item.content["emergency_type"]].append(item)

        # Analyze each emergency type
        for emergency_type, items in by_emergency_type.items():
            # Extract generation times for this emergency type
            generation_times = []
            for item in items:
                if "generation_time" in item.content:
                    generation_times.append(item.content["generation_time"])

            if generation_times:
                avg_time = sum(generation_times) / len(generation_times)
                insight_id = f"insight_emergency_{emergency_type}_{timestamp.strftime('%Y%m%d%H%M%S')}"
                
                # For emergency mode, even slightly elevated times are concerning
                if avg_time > 5.0:  # Lower threshold for emergency mode
                    description = f"High generation time of {avg_time:.2f} seconds in {emergency_type} emergency mode"
                    actions = [
                        "Optimize emergency mode pipeline",
                        "Consider more aggressive quality-speed tradeoffs for this emergency type",
                        "Review template complexity for this emergency type"
                    ]
                    priority = 4  # Higher priority for emergency performance issues
                else:
                    description = f"Acceptable generation time of {avg_time:.2f} seconds in {emergency_type} emergency mode"
                    actions = ["Continue monitoring emergency performance"]
                    priority = 2  # Still moderate priority for emergency monitoring
                
                insight = FeedbackInsight(
                    id=insight_id,
                    timestamp=timestamp,
                    metric=FeedbackMetric.EMERGENCY_PERFORMANCE,
                    value=avg_time,
                    confidence=0.85,
                    source_feedback_ids=[item.id for item in items if "generation_time" in item.content],
                    description=description,
                    suggested_actions=actions,
                    tags=["emergency", emergency_type, "performance"],
                    priority=priority
                )
                
                insights.append(insight)

        return insights

    def _analyze_content_accuracy(self, feedback_items: List[FeedbackItem]) -> List[FeedbackInsight]:
        """Analyze content accuracy feedback to generate insights.

        Args:
            feedback_items: List of accuracy feedback items

        Returns:
            List of generated insights
        """
        insights = []
        timestamp = datetime.datetime.now()

        if not feedback_items:
            return insights

        # Count accuracy issues by type
        issue_types = Counter()
        for item in feedback_items:
            if "issue_type" in item.content:
                issue_types[item.content["issue_type"]] += 1

        # Generate insights for common issue types
        for issue_type, count in issue_types.most_common(3):  # Focus on top 3 issues
            if count >= 3:  # Only if we have at least 3 occurrences
                insight_id = f"insight_accuracy_{issue_type}_{timestamp.strftime('%Y%m%d%H%M%S')}"
                
                description = f"Frequent accuracy issue: {issue_type} (occurred {count} times)"
                
                # Suggest actions based on issue type
                if issue_type == "factual_error":
                    actions = [
                        "Enhance fact-checking module",
                        "Consider additional knowledge sources",
                        "Implement stricter verification for factual content"
                    ]
                elif issue_type == "visual_mismatch":
                    actions = [
                        "Improve text-to-image alignment",
                        "Review visual consistency checks",
                        "Consider additional training for visual coherence"
                    ]
                elif issue_type == "temporal_inconsistency":
                    actions = [
                        "Enhance temporal coherence module",
                        "Review frame-to-frame consistency algorithms",
                        "Consider additional training for temporal stability"
                    ]
                else:
                    actions = [
                        f"Investigate root causes of {issue_type} issues",
                        "Develop targeted improvements for this issue type"
                    ]
                
                insight = FeedbackInsight(
                    id=insight_id,
                    timestamp=timestamp,
                    metric=FeedbackMetric.QUALITY_SCORE,
                    value=count,  # Using count as a proxy for severity
                    confidence=0.75,
                    source_feedback_ids=[item.id for item in feedback_items 
                                        if "issue_type" in item.content and item.content["issue_type"] == issue_type],
                    description=description,
                    suggested_actions=actions,
                    tags=["accuracy", issue_type],
                    priority=3  # Accuracy issues are important
                )
                
                insights.append(insight)

        return insights

    def get_insights(self, 
                    start_date: Optional[datetime.datetime] = None,
                    end_date: Optional[datetime.datetime] = None,
                    metric: Optional[FeedbackMetric] = None,
                    min_confidence: float = 0.0,
                    tags: Optional[List[str]] = None,
                    min_priority: int = 0) -> List[FeedbackInsight]:
        """Get insights with optional filtering.

        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            metric: Optional metric filter
            min_confidence: Minimum confidence threshold
            tags: Optional list of tags to filter by (any match)
            min_priority: Minimum priority level

        Returns:
            List of insights matching the criteria
        """
        filtered_insights = []

        for insight in self.insights:
            # Apply filters
            if start_date and insight.timestamp < start_date:
                continue
            if end_date and insight.timestamp > end_date:
                continue
            if metric and insight.metric != metric:
                continue
            if insight.confidence < min_confidence:
                continue
            if tags and not any(tag in insight.tags for tag in tags):
                continue
            if insight.priority < min_priority:
                continue
                
            filtered_insights.append(insight)

        # Sort by priority (descending) and then by timestamp (descending)
        filtered_insights.sort(key=lambda x: (-x.priority, -x.timestamp.timestamp()))
        
        return filtered_insights

    def get_top_insights(self, limit: int = 5) -> List[FeedbackInsight]:
        """Get the top insights based on priority and recency.

        Args:
            limit: Maximum number of insights to return

        Returns:
            List of top insights
        """
        # Sort by priority (descending) and then by timestamp (descending)
        sorted_insights = sorted(
            self.insights, 
            key=lambda x: (-x.priority, -x.timestamp.timestamp())
        )
        
        return sorted_insights[:limit]

    def get_trend_analysis(self, 
                          metric: FeedbackMetric,
                          time_window_days: int = 30,
                          interval_days: int = 1) -> Dict[str, Any]:
        """Analyze trends for a specific metric over time.

        Args:
            metric: The metric to analyze
            time_window_days: Number of days to look back
            interval_days: Interval size in days

        Returns:
            Dictionary with trend analysis results
        """
        # Calculate date range
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=time_window_days)
        
        # Filter insights by metric and date range
        relevant_insights = [
            insight for insight in self.insights
            if insight.metric == metric and start_date <= insight.timestamp <= end_date
        ]
        
        if not relevant_insights:
            return {
                "metric": metric.name,
                "time_window_days": time_window_days,
                "interval_days": interval_days,
                "data_points": [],
                "trend": "insufficient_data",
                "analysis": "Insufficient data for trend analysis"
            }
        
        # Group insights by time intervals
        intervals = []
        current_date = start_date
        while current_date <= end_date:
            interval_end = current_date + datetime.timedelta(days=interval_days)
            interval_insights = [
                insight for insight in relevant_insights
                if current_date <= insight.timestamp < interval_end
            ]
            
            if interval_insights:
                avg_value = sum(insight.value for insight in interval_insights) / len(interval_insights)
                intervals.append({
                    "start_date": current_date.isoformat(),
                    "end_date": interval_end.isoformat(),
                    "value": avg_value,
                    "count": len(interval_insights)
                })
            
            current_date = interval_end
        
        # Calculate trend
        if len(intervals) >= 2:
            first_value = intervals[0]["value"]
            last_value = intervals[-1]["value"]
            change = last_value - first_value
            percent_change = (change / first_value) * 100 if first_value != 0 else 0
            
            if percent_change > 10:
                trend = "significant_increase"
                analysis = f"The {metric.name} has increased significantly by {percent_change:.1f}% over the time period."
            elif percent_change > 2:
                trend = "slight_increase"
                analysis = f"The {metric.name} has slightly increased by {percent_change:.1f}% over the time period."
            elif percent_change < -10:
                trend = "significant_decrease"
                analysis = f"The {metric.name} has decreased significantly by {abs(percent_change):.1f}% over the time period."
            elif percent_change < -2:
                trend = "slight_decrease"
                analysis = f"The {metric.name} has slightly decreased by {abs(percent_change):.1f}% over the time period."
            else:
                trend = "stable"
                analysis = f"The {metric.name} has remained relatively stable over the time period."
        else:
            trend = "insufficient_data"
            analysis = "Insufficient data points for trend analysis"
        
        return {
            "metric": metric.name,
            "time_window_days": time_window_days,
            "interval_days": interval_days,
            "data_points": intervals,
            "trend": trend,
            "analysis": analysis
        }