"""
Core algorithms for TI4 map generation and optimization.
"""

from .hex_grid import HexCoord, hex_distance
from .balance_engine import (
    TI4Map,
    HomeValue,
    improve_balance,
    analyze_balance,
    get_home_values,
    get_balance_gap
)
from .map_generator import (
    generate_random_map,
    generate_multiple_maps,
    get_map_statistics
)
from .spatial_optimizer import (
    MultiObjectiveScore,
    evaluate_map_multiobjective,
    improve_balance_spatial,
    pareto_optimize,
    compare_optimizers
)

__all__ = [
    # hex_grid
    'HexCoord',
    'hex_distance',
    # balance_engine
    'TI4Map',
    'HomeValue',
    'improve_balance',
    'analyze_balance',
    'get_home_values',
    'get_balance_gap',
    # map_generator
    'generate_random_map',
    'generate_multiple_maps',
    'get_map_statistics',
    # spatial_optimizer
    'MultiObjectiveScore',
    'evaluate_map_multiobjective',
    'improve_balance_spatial',
    'pareto_optimize',
    'compare_optimizers',
]
