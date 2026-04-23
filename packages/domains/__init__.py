"""
Heli.OS Domain Logic Layer

Domain modules sit above adapters and below the mission orchestrator.
Each module provides:
  - Domain-specific situation assessment (risk scoring, pattern recognition)
  - Mission type recommendations based on domain context
  - Reporting format selection
  - Asset capability requirements for domain operations

Available domains:
  - maritime:     Port security, vessel ops, maritime SAR
  - military:     HADR, ACE, force protection, CASEVAC escort
  - utilities:    Power line, pipeline, bridge inspection
  - agriculture:  Crop monitoring, precision ag, livestock
  - oilgas:       Pipeline patrol, flare monitoring, spill detection
  - construction: Site monitoring, safety compliance, progress tracking
  - wildlife:     Anti-poaching, species tracking, habitat monitoring
"""

from .maritime import assess_maritime_situation, plan_maritime_mission
from .military import assess_military_situation, plan_military_mission
from .utilities import assess_utilities_situation, plan_utilities_mission
from .agriculture import assess_agriculture_situation, plan_agriculture_mission
from .oilgas import assess_oilgas_situation, plan_oilgas_mission
from .construction import assess_construction_situation, plan_construction_mission
from .wildlife import assess_wildlife_situation, plan_wildlife_mission

__all__ = [
    "assess_maritime_situation",
    "plan_maritime_mission",
    "assess_military_situation",
    "plan_military_mission",
    "assess_utilities_situation",
    "plan_utilities_mission",
    "assess_agriculture_situation",
    "plan_agriculture_mission",
    "assess_oilgas_situation",
    "plan_oilgas_mission",
    "assess_construction_situation",
    "plan_construction_mission",
    "assess_wildlife_situation",
    "plan_wildlife_mission",
]
