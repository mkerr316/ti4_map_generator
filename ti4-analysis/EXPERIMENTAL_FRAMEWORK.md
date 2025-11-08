# Spatial Blindness Experimental Framework

## Overview

This framework implements a rigorous experimental design to test the central hypothesis from the ti4-analysis research:

> **A map generator optimizing *only* for "basic" balance metrics (like `balance_gap`) is "spatially blind" to critical spatial imbalances.**

## Quick Start

### 1. Install Dependencies

```bash
cd ti4-analysis
pip install -r requirements.txt
```

### 2. Run Validation Test

```bash
python tests/test_pipeline.py
```

This runs a quick validation (~2 minutes) to ensure all components work.

### 3. Run Quick Experiment (N=10)

```bash
python ../experiments/run_spatial_blindness_experiment.py --quick
```

Estimated time: ~5 minutes

### 4. View Results

Results are saved to `ti4-analysis/results/spatial_blindness_TIMESTAMP/`:
- **`SPATIAL_BLINDNESS_REPORT.md`** - Main findings and analysis
- **`figures/`** - All visualizations (PNG)
- **`raw_data/`** - CSV files with all metrics

## Experimental Design

### Hypothesis

The "basic" balance optimizer (which minimizes `balance_gap` through iterative tile swapping) achieves excellent balance in raw resource values but **ignores spatial distribution patterns**, potentially creating maps that are:

- **Perfectly balanced** in total home value
- **Spatially imbalanced** with clustered resources or unequal accessibility

### Methodology

For each of N random maps:

1. **Generate** a naive (random) map
2. **Measure** initial balance and spatial metrics
3. **Optimize** using the basic balance algorithm (`improve_balance`)
4. **Measure** final balance and spatial metrics
5. **Compare** distributions and correlations

### Metrics Collected

**Basic Balance Metrics:**
- `balance_gap` - Max-min difference in home values
- `mean` - Average home value
- `std` - Standard deviation of home values
- `fairness_index` - Jain's fairness index for values

**Spatial Distribution Metrics:**
- `morans_i` - Spatial autocorrelation (resource clustering)
- `jains_fairness_index` - Accessibility fairness
- `gini_coefficient` - Accessibility inequality
- `num_hotspots` / `num_coldspots` - Statistical spatial clusters
- `accessibility_variance` - Variance in distance-weighted accessibility

### Statistical Tests

1. **Paired t-tests** - Test if optimization changes each metric
   - Expected: Balance gap decreases (optimizer works)
   - Expected: Spatial metrics DON'T change (spatial blindness)

2. **Correlation analysis** - Test independence of objectives
   - Expected: Weak correlation between balance_gap and spatial metrics
   - Expected: Non-significant p-values (proves independence)

3. **Smoking gun cases** - Find maps with perfect balance but poor spatial distribution

## Usage Examples

### Standard Experiments

```bash
# Quick validation (N=10, ~5 min)
python ../experiments/run_spatial_blindness_experiment.py --quick

# Medium scale (N=50, ~25 min)
python ../experiments/run_spatial_blindness_experiment.py --sample-sizes 50

# Multi-scale (N=10,50,200, ~2 hours)
python ../experiments/run_spatial_blindness_experiment.py --sample-sizes 10 50 200

# Full publication quality (N=1000, ~8 hours)
python ../experiments/run_spatial_blindness_experiment.py --full
```

### Custom Configuration

```bash
# Custom sample sizes
python ../experiments/run_spatial_blindness_experiment.py --sample-sizes 25 100 500

# More optimization iterations
python ../experiments/run_spatial_blindness_experiment.py --quick --iterations 300

# Different player count
python ../experiments/run_spatial_blindness_experiment.py --quick --player-count 4

# Custom output directory
python ../experiments/run_spatial_blindness_experiment.py --quick --output-dir ./my_results

# Base game only (no PoK)
python ../experiments/run_spatial_blindness_experiment.py --quick --no-pok
```

### Programmatic Usage

```python
from ti4_analysis.experiments import run_batch_experiment, analyze_experiment_results
from ti4_analysis.visualization import create_all_experiment_visualizations

# Run experiment
df = run_batch_experiment(
    num_maps=50,
    player_count=6,
    optimization_iterations=200,
    base_seed=42
)

# Analyze results
paired_results = analyze_experiment_results(df)
correlation_results = test_spatial_blindness(df)

# Generate visualizations
figure_paths = create_all_experiment_visualizations(
    df, paired_results, correlation_results, output_dir="./results"
)
```

## Architecture

### Components

```
ti4-analysis/
├── src/ti4_analysis/
│   ├── data/
│   │   ├── tile_loader.py          # Parse JavaScript tile database
│   │   └── map_structures.py       # Data structures (System, Planet, etc.)
│   ├── algorithms/
│   │   ├── balance_engine.py       # Basic balance optimizer
│   │   ├── map_generator.py        # Random map generation
│   │   └── spatial_optimizer.py    # Multi-objective optimizer (NEW!)
│   ├── spatial_stats/
│   │   └── spatial_metrics.py      # Moran's I, Jain's Index, etc.
│   ├── visualization/
│   │   ├── map_viz.py             # Map plotting functions
│   │   └── experiment_viz.py      # Experimental visualizations (NEW!)
│   └── experiments/
│       ├── batch_experiment.py    # Batch experiment runner (NEW!)
│       ├── analysis.py            # Statistical tests (NEW!)
│       └── report_generator.py    # Markdown reports (NEW!)
├── experiments/
│   └── run_spatial_blindness_experiment.py  # Main orchestration script
├── tests/
│   └── test_pipeline.py           # Validation tests
└── results/                       # Experimental outputs
```

### Key Classes

**`TI4Map`** - Represents a game map with hex spaces
- `get_home_spaces()` - Get player starting positions
- `get_system_spaces()` - Get all tile spaces
- `copy()` - Deep copy for non-destructive optimization

**`Evaluator`** - Tile value evaluator (e.g., "Joebrew")
- Configurable weights for resources, influence, tech specs, etc.
- Distance multipliers for adjacency-based evaluation

**`TileDatabase`** - Complete tile pool
- Base game + expansions (PoK, Uncharted, etc.)
- Categorized by type (blue/red/home)
- `get_swappable_tiles()` - Get random tile pool

**`MultiObjectiveScore`** - Multi-objective fitness
- Tracks balance_gap, morans_i, jains_index
- `composite_score()` - Weighted aggregate
- `dominates()` - Pareto dominance check

## Advanced Features

### Multi-Objective Spatial Optimizer

The framework includes a multi-objective optimizer that balances **both** basic balance AND spatial distribution:

```python
from ti4_analysis.algorithms import improve_balance_spatial, compare_optimizers

# Optimize with spatial awareness
final_score, history = improve_balance_spatial(
    ti4_map,
    evaluator,
    iterations=200,
    weights={
        'balance_gap': 1.0,       # Minimize gap
        'morans_i': 0.5,          # Minimize clustering
        'jains_index': -0.5       # Maximize fairness
    }
)

# Compare basic vs spatial
results = compare_optimizers(ti4_map, evaluator)
# Prints comparison table
```

### Pareto Optimization

Find the Pareto-optimal front of maps with different trade-offs:

```python
from ti4_analysis.algorithms import pareto_optimize

# Find Pareto front (computationally expensive!)
pareto_front = pareto_optimize(
    ti4_map,
    evaluator,
    iterations=100,
    population_size=20
)

# pareto_front is a list of (map, score) tuples
# Each represents a non-dominated solution
```

## Output Files

### Results Directory Structure

```
results/spatial_blindness_YYYYMMDD_HHMMSS/
├── n0010/                         # N=10 experiment
│   ├── raw_data/
│   │   └── results_n0010.csv     # All metrics for 10 maps
│   ├── figures/
│   │   ├── multi_metric_comparison.png
│   │   ├── distribution_balance_gap.png
│   │   ├── scatter_gap_vs_morans.png
│   │   ├── scatter_gap_vs_jains.png
│   │   ├── effect_sizes.png
│   │   └── convergence_comparison.png
│   └── SPATIAL_BLINDNESS_REPORT.md
├── n0050/                         # N=50 experiment
│   └── ...
└── n0200/                         # N=200 experiment
    └── ...
```

### Report Contents

The auto-generated markdown report includes:

1. **Executive Summary** - Key findings and hypothesis test results
2. **Statistical Analysis** - Paired t-tests and correlations
3. **Visualizations** - Embedded PNG images
4. **Smoking Gun Cases** - Maps proving spatial blindness
5. **Conclusions** - Interpretation of results
6. **Recommendations** - Next steps for implementation

## Interpretation Guide

### Statistical Significance

- **p < 0.05**: Statistically significant (marked with ✓)
- **p ≥ 0.05**: Not significant

### Effect Sizes (Cohen's d)

- **|d| < 0.2**: Negligible
- **0.2 ≤ |d| < 0.5**: Small
- **0.5 ≤ |d| < 0.8**: Medium
- **|d| ≥ 0.8**: Large

### Correlation Strength

- **|r| < 0.3**: Weak (supports spatial blindness)
- **0.3 ≤ |r| < 0.7**: Moderate
- **|r| ≥ 0.7**: Strong

### Expected Results

If the hypothesis is correct:

✓ **Balance Gap**: Large negative Δ, p < 0.001, large effect size
✓ **Moran's I**: Small |Δ|, p > 0.05, negligible effect size
✓ **Jain's Index**: Small |Δ|, p > 0.05, negligible effect size
✓ **Correlations**: Weak |r| < 0.3, p > 0.05
✓ **Smoking Guns**: Multiple cases with gap < 1.0 but morans_i > 0.3

## Performance Notes

### Computational Cost

- **Per map**: ~30 seconds (200 iterations)
  - Map generation: ~1s
  - Optimization: ~20s
  - Metrics calculation: ~8s
  - Visualization: ~1s (if requested)

- **Batch experiments**:
  - N=10: ~5 minutes
  - N=50: ~25 minutes
  - N=200: ~2 hours
  - N=1000: ~8 hours

### Optimization

- Results are cached (tile database loaded once)
- Intermediate results saved (crash recovery)
- Parallel processing not implemented (future work)

## Troubleshooting

### Import Errors

```bash
# Ensure package is installed
cd ti4-analysis
pip install -e .
```

### Tile Database Not Found

The tile loader automatically searches parent directories for `src/data/tileData.js`. If it fails:

```python
from ti4_analysis.data import load_tile_database
from pathlib import Path

# Specify project root explicitly
project_root = Path("/path/to/ti4_map_generator")
db = load_tile_database(project_root=project_root)
```

### Memory Issues

For very large experiments (N>1000), consider:
- Running in batches
- Reducing `optimization_iterations`
- Disabling intermediate saves

### Visualization Errors

If matplotlib fails:
```bash
pip install --upgrade matplotlib seaborn
```

## Next Steps

### Recommended Workflow

1. **Run validation**: `python tests/test_pipeline.py`
2. **Quick experiment**: `--quick` (N=10, verify it works)
3. **Medium scale**: `--sample-sizes 50` (publishable results)
4. **Large scale**: `--sample-sizes 200 1000` (comprehensive analysis)

### Based on Results

If spatial blindness is confirmed:

1. **Implement multi-objective optimizer in main generator**
   - Use `spatial_optimizer.py` as reference
   - Add UI controls for weight preferences

2. **Update MapOptions.js**
   - Add toggle for "spatial balance mode"
   - Display Moran's I and Jain's Index in UI

3. **Validate with players**
   - Playtesting with spatial-optimized maps
   - Survey perceived fairness

4. **Publish findings**
   - Share report with TI4 community
   - Present at gaming conferences

## References

### Research Background

- [ti4-analysis README](README.md) - Background on balance metrics
- Original experimental design (provided by user)
- TI4 balance discussions on r/twilightimperium

### Spatial Statistics

- Moran's I: Moran, P. A. P. (1950). "Notes on continuous stochastic phenomena"
- Jain's Fairness Index: Jain, R. et al. (1984). "A quantitative measure of fairness"
- Getis-Ord Gi*: Ord, J. K. and Getis, A. (1995). "Local spatial autocorrelation statistics"

### Multi-Objective Optimization

- Pareto optimization: Coello, C. A. C. (2006). "Evolutionary multi-objective optimization"
- NSGA-II algorithm: Deb, K. et al. (2002). "A fast and elitist multi-objective genetic algorithm"

## License

Same as parent project (ti4_map_generator).

## Contributors

- Original ti4-analysis framework: [Original authors]
- Experimental framework: Claude Code implementation (2025)

## Citation

If you use this framework in research or publications:

```
@software{ti4_spatial_blindness_2025,
  title = {Spatial Blindness Experimental Framework for TI4 Map Generation},
  author = {TI4 Map Generator Team},
  year = {2025},
  url = {https://github.com/yourusername/ti4_map_generator}
}
```
