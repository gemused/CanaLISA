import os
import sys
import lisaglitch
import numpy as np
import ldc.io.yml as ymlio
import argparse


PATH_src = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PATH_bethLISA = os.path.abspath(os.path.join(PATH_src, os.pardir))
PATH_glitch_config = os.path.join(PATH_bethLISA, "dist/glitch_config/")
PATH_pipeline_config = os.path.join(PATH_bethLISA, "dist/pipeline_config/")
PATH_glitch_data = os.path.join(PATH_bethLISA, "dist/glitch_data/")
PATH_simulation_data = os.path.join(PATH_bethLISA, "dist/simulation_data/")
PATH_tdi_data = os.path.join(PATH_bethLISA, "dist/tdi_data/")
PATH_orbit_data = os.path.join(PATH_bethLISA, "dist/orbit_data/")


def init_cl():
    """Initialize commandline arguments and return Namespace object with all
    given commandline arguments.
    """

    parser = argparse.ArgumentParser()

    # FILE MANAGEMENT
    parser.add_argument(
        "--glitch_output_h5",
        type=str,
        default="default_glitch_output.h5",
        help="Glitch output h5 file name",
    )
    parser.add_argument(
        "--glitch_output_txt",
        type=str,
        default="default_glitch_output.txt",
        help="Glitch output txt file name",
    )
    parser.add_argument(
        "--glitch_cfg_input",
        type=str,
        help="Glitch config file name",
    )
    parser.add_argument(
        "--pipe_cfg_input",
        type=str,
        help="Pipeline config file name",
    )

    # GLITCH ARGUMENTS
    parser.add_argument(
        "--glitch_type",
        type=str,
        default="Poisson",
        help=""
    )
    parser.add_argument(
        "--amp_type",
        type=str,
        default="Gaussian",
        help=""
    )
    parser.add_argument(
        "--beta_type",
        type=str,
        default="Exponential",
        help=""
    )
    parser.add_argument(
        "--t_min",
        type=float,
        default=0.0,
        help=""
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=6307200.0,
        help=""
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=5,
        help=""
    )
    parser.add_argument(
        "--physic_upsampling",
        type=float,
        default=1.0,
        help=""
    )
    parser.add_argument(
        "--glitch_rate",
        type=int,
        default=0.2,
        help="Only used for glitch_type = poisson",
    )
    parser.add_argument(
        "--glitch_spacing",
        type=int,
        default=20000,
        help="Only used for glitch_type = equal_spacing",
    )
    parser.add_argument(
        "--avg_amp",
        type=float,
        default=10**-12,
        help=""
    )
    parser.add_argument(
        "--std_amp",
        type=float,
        default=10**-10,
        help=""
    )
    parser.add_argument(
        "--beta_scale",
        type=int,
        default=50,
        help=""
    )
    parser.add_argument(
        "--amp_set_min",
        type=float,
        default=10**-10,
        help=""
    )
    parser.add_argument(
        "--amp_set_max",
        type=float,
        default=10**-5,
        help=""
    )
    parser.add_argument(
        "--beta_set_min",
        type=float,
        default=0.001,
        help=""
    )
    parser.add_argument(
        "--beta_set_max",
        type=float,
        default=100,
        help=""
    )

    # SEED
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to ensure deterministic outputs"
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
        "t0": cl_args.t_min,
        "t_max": cl_args.t_max,
        "dt": cl_args.dt,
        "physic_upsampling": cl_args.physic_upsampling,
        "size": cl_args.t_max / cl_args.dt,
        "glitch_type": cl_args.glitch_type,
        "glitch_rate": cl_args.glitch_rate,
        "glitch_spacing": cl_args.glitch_spacing,
        "amp_type": cl_args.amp_type,
        "avg_amp": cl_args.avg_amp,
        "std_amp": cl_args.std_amp,
        "amp_set": [cl_args.amp_set_min, cl_args.amp_set_max],
        "beta_type": cl_args.beta_type,
        "beta_set": [cl_args.beta_set_min, cl_args.beta_set_max],
        "beta_scale": cl_args.beta_scale,
        "glitch_output_h5": PATH_glitch_data + cl_args.glitch_output_h5,
        "glitch_output_txt": PATH_glitch_data + cl_args.glitch_output_txt,
        "seed": cl_args.seed,
    }

    return params


def file_paths_to_params(
    glitch_cfg_path, pipe_cfg_path, glitch_output_h5, glitch_output_txt
):
    """Returns a dictionary of all the needed parameters (and combinations of
    parameters) from glitch_cfg and pipe_cfg files.

    Arguments:
      glitch_cfg_path -- path to glitch_cfg file
      pipe_cfg_path -- path to pipe_cfg file
      glitch_output_h5 -- path to final glitch file in as .h5
      glitch_output_txt -- path to final glitch file in as .txt
    """

    glitch_cfg = ymlio.load_config(glitch_cfg_path)
    pipe_cfg = ymlio.load_config(pipe_cfg_path)
    t0 = 10368000

    if "glitch_0" in glitch_cfg:
        glitches = [value for key, value in glitch_cfg.items()]

        for glitch in glitches:
            glitch["t_inj"] += t0
    else:
        glitches = make_glitches(glitch_cfg, pipe_cfg, t0)

    params = {
        "t0": t0,
        "size": pipe_cfg["duration"].to("s").value
        / pipe_cfg["dt"].to("s").value,
        "dt": pipe_cfg["dt"].to("s").value
        / pipe_cfg["physic_upsampling"],
        "physic_upsampling": pipe_cfg["physic_upsampling"],
        "glitches": glitches,
        "glitch_output_h5": PATH_glitch_data + glitch_output_h5,
        "glitch_output_txt": PATH_glitch_data + glitch_output_txt,
    }

    return params


def make_glitches(glitch_cfg, pipe_cfg, t0):
    glitches = []
    t_injs = []
    window = 500
    t_max = t0 + pipe_cfg["duration"].to("s").value

    for i in range(int(glitch_cfg["daily_rate"] * pipe_cfg["duration"].to("d").value)):
        invalid_t_inj = True
        while invalid_t_inj:
            t_inj = np.random.randint(t0, t0 + pipe_cfg["duration"].to("s").value)
            if t_injs:
                for t in t_injs:
                    if t_inj < t + window and t_inj > t - window and t_inj + window < t_max:
                        invalid_t_inj = True
                        break
                    else:
                        invalid_t_inj = False
            else:
                invalid_t_inj = False
        t_injs.append(t_inj)

        glitch = {
            "shape": glitch_cfg["shape"],
            "inj_point": np.random.choice(glitch_cfg["inj_points"]),
            "t_inj": t_inj,
            "t_rise": np.random.randint(glitch_cfg["t_rise_range"][0],
                                        glitch_cfg["t_rise_range"][1]),
            "t_fall": np.random.randint(glitch_cfg["t_fall_range"][0],
                                        glitch_cfg["t_fall_range"][1]),
            "level": np.random.randint(float(glitch_cfg["level_range"][0]),
                                       float(glitch_cfg["level_range"][1])),
        }

        glitches.append(glitch)
    
    return glitches


def simulate_glitches(params):
    """Simulate glitches given dictionary of parameters and write glitches to
    file.

    Arguments:
    params -- dictionary of parameters describing glitches to simulate
    """
    np.random.seed(params["seed"])

    inj_points = ["tm_12"]

    glitches = params["glitches"]

    # PRODUCE GLITCHES
    if os.path.exists(params["glitch_output_h5"]):
        os.remove(params["glitch_output_h5"])

    glitch_list = []
    for glitch in glitches:
        if glitch["shape"] == "IntegratedShapeletGlitch":
            g = lisaglitch.IntegratedShapeletGlitch(
                inj_point=np.random.choice(inj_points),
                t0=params["t0"],
                size=params["size"],
                dt=params["dt"],
                t_inj=glitch["t_inj"],
                beta=glitch["beta"],
                level=glitch["level"],
            )
        elif glitch["shape"] == "OneSidedDoubleExpGlitch":
            g = lisaglitch.OneSidedDoubleExpGlitch(
                inj_point=glitch["inj_point"],
                t0=params["t0"],
                size=params["size"],
                dt=params["dt"],
                t_inj=glitch["t_inj"],
                t_rise=glitch["t_rise"],
                t_fall=glitch["t_fall"],
                level=glitch["level"]
            )
        else:
            sys.exit("Not a supported glitch type")

        glitch_list.append(g)
        g.write(path=params["glitch_output_h5"], mode="a")

    # FORMAT/MAKE GLITCH FILE
    header = (
        "shape "
        + "inj_point "
        + "size  "
        + "dt  "
        + "physics_upsampling  "
        + "t0  "
        + "t_inj "
        + "level "
        + "t_rise "
        + "t_fall "
    )

    output_txt = params["glitch_output_txt"]

    if os.path.exists(output_txt):
        os.remove(output_txt)

    with open(f"{output_txt}", "w") as f:
        f.write(header + "\n")

    with open(f"{output_txt}", "a") as f:
        for glitch in glitches:
            f.write(
                glitch["shape"] + "  "
                + str(glitch["inj_point"]) + " "
                + str(params["size"]) + "  "
                + str(params["dt"]) + "  "
                + str(params["physic_upsampling"]) + "  "
                + str(params["t0"]) + "  "
                + str(glitch["t_inj"]) + " "
                + str(float(glitch["level"])) + " "
                + str(glitch["t_rise"]) + " "
                + str(glitch["t_fall"]) + "\n"
            )


def make_glitch(args):
    if args is not None:
        params = file_paths_to_params(
            PATH_glitch_config + args.glitch_cfg_input,
            PATH_pipeline_config + args.pipe_cfg_input,
            args.glitch_output_h5,
            args.glitch_output_txt,
        )
        params["seed"] = args.seed
    else:
        cl_args = init_cl()
        if cl_args.glitch_cfg_input is not None \
                and cl_args.pipe_cfg_input is not None:
            params = file_paths_to_params(
                PATH_glitch_config + cl_args.glitch_cfg_input,
                PATH_pipeline_config + cl_args.pipe_cfg_input,
                cl_args.glitch_output_h5,
                cl_args.glitch_output_txt,
            )
            params["seed"] = cl_args.seed
        else:
            params = cl_args_to_params(cl_args)

    simulate_glitches(params)


"""Uncomment to run make_glitch alone"""
# if __name__ == "__main__":
#     make_glitch(args=None)
