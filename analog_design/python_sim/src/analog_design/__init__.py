"""Analog design simulation package."""

from .simulator import (
    Circuit,
    BSIM4DCAnalyzer,
    BSIM4TRANAnalyzer,
    parse_netlist,
    compare_with_golden,
    VSource,
    ISource,
    Resistor,
    Capacitor,
    MOSFET,
)
from .plotting import (
    plot_tran_compare,
    plot_comparator_waveforms,
)

__all__ = [
    "Circuit",
    "BSIM4DCAnalyzer",
    "BSIM4TRANAnalyzer",
    "parse_netlist",
    "compare_with_golden",
    "VSource",
    "ISource",
    "Resistor",
    "Capacitor",
    "MOSFET",
    "plot_tran_compare",
    "plot_comparator_waveforms",
]
