import os
import argparse
import ldc.io.yml as ymlio
from lisagwresponse import VerificationBinary

PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_glitch_config = os.path.join(PATH_bethLISA, "dist/glitch_config/")
PATH_pipeline_config = os.path.join(PATH_bethLISA, "dist/pipeline_config/")
PATH_glitch_data = os.path.join(PATH_bethLISA, "dist/glitch_data/")
PATH_simulation_data = os.path.join(PATH_bethLISA, "dist/simulation_data/")
PATH_tdi_data = os.path.join(PATH_bethLISA, "dist/tdi_data/")
PATH_orbit_data = os.path.join(PATH_bethLISA, "dist/orbit_data/")
PATH_gw_signal_config = os.path.join(PATH_bethLISA, "dist/gw_signal_config/")
PATH_gw_signal_data = os.path.join(PATH_bethLISA, "dist/gw_signal_data/")


def init_cl():
    """Initialize commandline arguments and return Namespace object with all
    given commandline arguments.
    """

    parser = argparse.ArgumentParser()

    # FILE MANAGEMENT
    parser.add_argument(
        "--gw_signal_cfg_input",
        type=str,
        help="Glitch config file name",
    )
    parser.add_argument(
        "--pipe_cfg_input",
        type=str,
        help="Pipeline config file name",
    )
    parser.add_argument(
        "--gw_signal_output_h5",
        type=str,
        default="default_gw_signal_output.h5",
        help="Glitch output h5 file name",
    )
    parser.add_argument(
        "--orbit_input_h5",
        type=str,
        default="orbits.h5",
        help="Orbit .h5 file name",
    )

    # GW SIGNAL ARGUMENTS
    parser.add_argument(
        "--period",
        type=float,
        help="System period (s)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        help="Luminosity distance (pc)",
    )
    parser.add_argument(
        "--mass1",
        type=float,
        help="Mass of object 1 (solar masses)",
    )
    parser.add_argument(
        "--mass2",
        type=float,
        help="Mass of object 2 (solar masses)",
    )
    parser.add_argument(
        "--glong",
        type=float,
        help="Galatic longitude (deg)",
    )
    parser.add_argument(
        "--glat",
        type=float,
        help="Galatic latitude (deg)",
    )
    parser.add_argument(
        "--iota",
        type=float,
        help="Inclination angle (rad)",
    )
    parser.add_argument(
        "--size",
        type=float,
        help="",
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="",
    )
    parser.add_argument(
        "--t0",
        type=float,
        help="",
    )

    return parser.parse_args()


def cl_args_to_params(cl_args):
    """Returns a dictionary of all the needed parameters (and combinations of
    parameters) from cl_args.

    Arguments:
    cl_args -- Namespace object with all parameters given in commandline
    arguments
    """

    params = {
        "period": cl_args.period,
        "distance": cl_args.distance,
        "masses": (cl_args.mass1, cl_args.mass2),
        "glong": cl_args.glong,
        "glat": cl_args.glat,
        "iota": cl_args.iota,
        "orbits": PATH_orbit_data + cl_args.orbit_input_h5,
        "size": cl_args.size,
        "dt": cl_args.dt,
        "t0": cl_args.t0,
        "gw_signal_output_h5": PATH_gw_signal_data
        + cl_args.gw_signal_output_h5,
    }

    return params


def file_paths_to_params(
    gw_signal_cfg_path, pipe_cfg_path, gw_signal_output_h5, orbit_input_h5
):
    """Returns a dictionary of all the needed parameters (and combinations of
    parameters) from glitch_cfg and pipe_cfg files.

    Arguments:
      gw_signal_cfg_path -- path to gw_signal_cfg file
    """

    gw_signal_cfg = ymlio.load_config(gw_signal_cfg_path)
    pipe_cfg = ymlio.load_config(pipe_cfg_path)

    params = {
        "period": gw_signal_cfg["period"],
        "distance": gw_signal_cfg["distance"],
        "masses": (gw_signal_cfg["mass1"], gw_signal_cfg["mass2"]),
        "glong": gw_signal_cfg["glong"],
        "glat": gw_signal_cfg["glat"],
        "iota": gw_signal_cfg["iota"],
        "orbits": PATH_orbit_data + orbit_input_h5,
        "size": gw_signal_cfg["size"],
        "dt": pipe_cfg["dt"],
        "t0": gw_signal_cfg["t0"],
        "gw_signal_output_h5": PATH_gw_signal_data + gw_signal_output_h5,
    }

    return params


def simulation_gw_signal(params):
    """function specification here please
    """
    source = VerificationBinary(
        period=params["period"].to("s").value,
        distance=params["distance"],
        masses=params["masses"],
        glong=params["glong"],
        glat=params["glat"],
        iota=params["iota"],
        orbits=params["orbits"],
        size=params["size"].to("s").value,
        dt=params["dt"].to("s").value,
        t0=params["t0"].to("s").value,
    )

    source.write(params["gw_signal_output_h5"])


def make_gw_signal(args):
    if args is not None:
        params = file_paths_to_params(
            PATH_gw_signal_config + args.gw_signal_cfg_input,
            PATH_pipeline_config + args.pipe_cfg_input,
            args.gw_signal_output_h5,
            args.orbit_input_h5,
        )
    else:
        cl_args = init_cl()
        if cl_args.gw_signal_output_h5 is not None:
            params = file_paths_to_params(
                PATH_gw_signal_config + cl_args.gw_signal_cfg_input,
                PATH_pipeline_config + cl_args.pipe_cfg_input,
                cl_args.gw_signal_output_h5,
                cl_args.orbit_input_h5,
            )
        else:
            params = cl_args_to_params(cl_args)

    simulation_gw_signal(params)


"""Uncomment to run make_gw_signal alone"""
# if __name__ == "__main__":
#     make_gw_signal(args=None)
