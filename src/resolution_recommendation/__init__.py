"""Resolution Recommendations Analysis Module.

Provides tools for extracting, clustering, and analyzing customer resolution
recommendations to identify actionable improvement themes.
"""

from .rr_analyzer import ResolutionRecommendationAnalyzer
from .RR_topic_analysis import RRtopicAnalyzer
__all__ = [
    "ResolutionRecommendationAnalyzer",
    "RRtopicAnalyzer"
]

__version__ = "1.0.0"
