# Diana-Sim: Analog Circuit Simulation with Graph Neural Networks

A comprehensive framework for analog circuit design and optimization using Graph Neural Networks (GNNs).

## Project Overview

Diana-Sim integrates graph-based representations of analog circuits with deep learning models to enable efficient circuit analysis, optimization, and design space exploration. The project focuses on DC analysis, graph data generation, and neural network-based circuit modeling.

## Project Structure

```
saratoga/
├── dc/                    # DC Analysis Module
│   ├── config.yaml                       # Main configuration file
│   ├── config_gnn_initial.yaml          # GNN initial configuration
│   ├── config_gnn_training.yaml         # GNN training configuration
│   ├── debug_dc_analysis.py             # DC analysis debugging script
│   └── plot_vout_model_comparison.py    # Model output comparison plotting
│
├── gdata/                 # Graph Data Generation
│   └── graph_gen.py                      # Graph generation and preprocessing script
│
└── gcn5/                  # Graph Convolutional Network Model
    ├── config.yaml                       # GCN configuration
    ├── config_utils.py                   # Configuration utility functions
    ├── gcn_dataset.py                    # Dataset handling and preparation
    ├── gcn_model.py                      # GCN model architecture
    ├── train_gcn.py                      # Training script
    └── infer_gcn.py                      # Inference/prediction script
```

## Module Descriptions

### DC Module (`dc/`)
Handles DC (Direct Current) circuit analysis and simulation:
- **dc_analysis.py**: Core DC analysis algorithms
- **plot_vout_*.py**: Visualization tools for output voltage analysis
- **config_*.yaml**: Configuration files for different analysis scenarios

### Graph Data Module (`gdata/`)
Generates and preprocesses graph representations of circuits:
- **graph_gen.py**: Converts circuit netlists to graph structures suitable for GNN processing
- Supports creation of node features, edge connections, and edge attributes

### GCN5 Module (`gcn5/`)
Graph Convolutional Network implementation for circuit modeling:
- **gcn_model.py**: GCN architecture with 5-layer depth
- **gcn_dataset.py**: PyTorch Dataset wrapper for graph data
- **train_gcn.py**: Training loop with validation and checkpoint saving
- **infer_gcn.py**: Inference pipeline for predictions on new circuits
- **config_utils.py**: Configuration loading and validation utilities

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric (PyG)
- PyYAML
- NumPy
- Matplotlib (for visualization)

## Installation

```bash
# Clone the repository
git clone https://github.com/ZijianHu2021/Diana-Sim.git
cd Diana-Sim

# Install dependencies
pip install torch torch-geometric pyyaml numpy matplotlib
```

## Quick Start

### 1. Generate Graph Data
```bash
cd gdata
python graph_gen.py
```

### 2. Train GCN Model
```bash
cd ../gcn5
python train_gcn.py --config config.yaml
```

### 3. Run Inference
```bash
python infer_gcn.py --config config.yaml --checkpoint <model.pt>
```

### 4. DC Analysis
```bash
cd ../dc
python debug_dc_analysis.py
```

## Configuration

Each module includes YAML configuration files. Key parameters:
- `dc/config.yaml`: Circuit parameters, simulation settings
- `gcn5/config.yaml`: Model hyperparameters (learning rate, batch size, epochs, etc.)

Modify these files to customize your experiments.

## Development

The codebase follows these principles:
- **Modularity**: Each component is independent and reusable
- **Configuration-Driven**: YAML configs separate hyperparameters from code
- **Reproducibility**: Seeds and configs ensure reproducible results

## Usage Examples

### Custom Circuit Analysis
```python
from dc.debug_dc_analysis import CircuitAnalyzer
analyzer = CircuitAnalyzer(config_path='dc/config.yaml')
results = analyzer.analyze(netlist_path='circuit.sp')
```

### GCN Prediction
```python
from gcn5.infer_gcn import GCNInference
model = GCNInference(checkpoint='models/gcn5.pt')
predictions = model.predict(graph_data)
```

## Output

- **Model checkpoints**: Saved during training for later inference
- **Analysis results**: DC analysis outputs as CSV/JSON
- **Visualizations**: Voltage curves and performance comparisons

## Author

Zijian Hu (Zijian.Hu@sony.com)

## License

This project is part of the Diana research initiative.

## Contributing

For contributions, please:
1. Create a feature branch
2. Make your changes
3. Ensure all tests pass
4. Submit a pull request

## References

- Graph Neural Networks for Circuit Design
- PyTorch Geometric Documentation
- Circuit Simulation and Optimization Techniques

---

**Last Updated**: February 5, 2026
