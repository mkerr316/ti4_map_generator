"""
Data structures and loaders for TI4 maps.
"""

from .map_structures import (
    Planet,
    System,
    MapSpace,
    MapSpaceType,
    Evaluator,
    PlanetTrait,
    TechSpecialty,
    PlanetEvalStrategy,
    Anomaly,
    Wormhole
)

from .tile_loader import (
    TileDatabase,
    load_tile_database,
    load_board_template
)

__all__ = [
    # map_structures
    'Planet',
    'System',
    'MapSpace',
    'MapSpaceType',
    'Evaluator',
    'PlanetTrait',
    'TechSpecialty',
    'PlanetEvalStrategy',
    'Anomaly',
    'Wormhole',
    # tile_loader
    'TileDatabase',
    'load_tile_database',
    'load_board_template',
]
