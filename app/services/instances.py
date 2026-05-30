"""
DataStorm 2026 - Service Instances
==================================
Initializes and holds singleton instances of all platform services.
Ensures zero circular imports and centralized lifecycle management.
"""

from app.services.db_service import DBService
from app.services.prediction_service import PredictionService
from app.services.spatial_service import SpatialService
from app.services.optimization_service import OptimizationService
from app.services.xai_service import XAIService

# Singleton instances
db = DBService()
prediction = PredictionService(db)
spatial = SpatialService(db)
optimization = OptimizationService(db)
xai = XAIService()
