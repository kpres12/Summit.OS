"""
Heli.OS Reporting Layer

Generates standardized operational reports from world model data.

Military / first-responder formats:
  - SALUTE / SPOT: enemy/unknown contact reports
  - 9-line MEDEVAC: casualty evacuation request format
  - SITREP: situation report

Infrastructure / commercial formats:
  - Inspection: utility, pipeline, bridge, construction inspection reports
"""

from .salute import generate_salute, generate_spot
from .nineline import generate_9line
from .sitrep import generate_sitrep
from .inspection import generate_inspection_report

__all__ = [
    "generate_salute",
    "generate_spot",
    "generate_9line",
    "generate_sitrep",
    "generate_inspection_report",
]
