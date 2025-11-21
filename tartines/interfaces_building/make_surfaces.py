import importlib
import interface_builder
import water_adder

# Reload the modules
importlib.reload(interface_builder)
importlib.reload(water_adder)

# Parameters
water_thickness = 15
intersubstrate_gap = 50
water_substrate_gap = 3
substrate_dir = 'substrates'

# Build the interface
interface_builder.interface_multi_builder(
    substrate_dir,
    water_thickness,
    intersubstrate_gap,
    water_substrate_gap,
    optimise_interface=True,
    logfile=True
)