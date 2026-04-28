"""Agent Improvements Analysis Module.

Provides tools for extracting, clustering, and analyzing agent performance
improvement opportunities to identify actionable training and development themes.
"""

from .ai_analyzer import AgentImprovementAnalyzer
from .ai_topic_analysis import AITopicAnalyzer

__all__ = [
    "AgentImprovementAnalyzer",
    "AITopicAnalyzer"
]

__version__ = "1.0.0"
