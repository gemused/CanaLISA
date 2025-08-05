"""
Filename: main.py
Author: William Mostrenko
Created: 2025-06-16
Description: Commandline organizer for LISA simulations.
"""

from inject_anomalies import inject_anomalies
import argparse
from make_anomalies import make_anomalies


def init_cl():
    """Initialize commandline arguments.

    Returns:
    Commandline arguments (namespace)
    """
    parser = argparse.ArgumentParser()

    # FILE MANAGEMENT
    parser.add_argument(
        "--glitch_cfg_input_fn",
        type=str,
        default="",
        help="Glitch config file name (excluding file extensions)"
    )
    parser.add_argument(
        "--gw_cfg_input_fn",
        type=str,
        default="",
        help="Gravitational wave config file name (excluding file extensions)"
    )
    parser.add_argument(
        "--pipe_cfg_input_fn",
        type=str,
        default="",
        help="Pipeline config file name (excluding file extensions)"
    )
    parser.add_argument(
        "--orbits_input_fn",
        type=str,
        default="orbits",
        help="Orbits file name (excluding file extensions)"
    )
    parser.add_argument(
        "--glitch_output_fn",
        type=str,
        default="default_glitch_output",
        help="Glitch output file name (excluding file extensions)"
    )
    parser.add_argument(
        "--gw_output_fn",
        type=str,
        default="default_gw_output",
        help="Gravitational wave output file name (excluding file extensions)"
    )
    parser.add_argument(
        "--pipe_output_fn",
        type=str,
        default="default_pipe_output",
        help="Pipeline output file name (excluding file extensions)"
    )
    parser.add_argument(
        "--anomaly_output_fn",
        type=str,
        default="default_anomaly_output",
        help="anomaly output file name (excluding file extensions)"
    )
    parser.add_argument(
        "--simulation_output_fn",
        type=str,
        default="default_simulation_output",
        help="Simulation output file name (excluding file extensions)"
    )
    parser.add_argument(
        "--tdi_output_fn",
        type=str,
        default="default_tdi_output",
        help="TDI output file name (excluding file extensions)"
    )

    # LISA INSTRUMENT ARGUMENTS
    parser.add_argument(
        "--disable_noise",
        type=bool,
        default=False,
        help="Simulate LISA instruments without noise?"
    )

    return parser.parse_args()


def main():
    """Simulates LISA with glitches and gws in specified config files."""
    args = init_cl()

    make_anomalies(
        glitch_cfg_input_fn=args.glitch_cfg_input_fn,
        gw_cfg_input_fn=args.gw_cfg_input_fn,
        pipe_cfg_input_fn=args.pipe_cfg_input_fn,
        orbits_input_fn=args.orbits_input_fn,
        glitch_output_fn=args.glitch_output_fn,
        gw_output_fn=args.gw_output_fn,
        pipe_output_fn=args.pipe_output_fn,
        anomaly_output_fn=args.anomaly_output_fn,
    )

    inject_anomalies(
        glitch_input_fn=args.glitch_output_fn,
        gw_input_fn=args.gw_output_fn,
        pipe_input_fn=args.pipe_output_fn,
        orbits_input_fn=args.orbits_input_fn,
        simulation_output_fn=args.simulation_output_fn,
        tdi_output_fn=args.tdi_output_fn,
        disable_noise=args.disable_noise,
    )


if __name__ == "__main__":
    main()
