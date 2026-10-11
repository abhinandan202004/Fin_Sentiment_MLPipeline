"""
Monitoring package for Fin_Sentiment_MLPipeline.
Exposes SystemHealthMonitor for operational telemetry and readiness diagnostics.
"""

from monitoring.system_health import SystemHealthMonitor

__all__ = ["SystemHealthMonitor"]
