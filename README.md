# Getting Started
## Simulating LISA Data
To run a simulation, from the command line navigate to `~/src/simulate_lisa/` and run `main.py`. An example is show below:
~~~
python main.py --glitch_cfg_input glitch_cfg --gw_cfg_input gw_cfg --pipe_cfg_input pipe_cfg 
~~~
The above command would run a LISA simulation with glitch parameters specified in `~/dist/glitch/glitch_config/glitch_cfg.yml`, gravitational wave parameters specified `~/dist/gw/gw_config/gw_cfg.yml`, and pipeline information specified in `~/dist/pipe/pipe_config/pipe_cfg.yml`.

The only parameters that you must specify for the simulation to run are those in the example above; `glitch_cfg_input_fn`, `gw_cfg_input_fn`, and `pipe_cfg_input_fn`. However, below is a list of the complete parameters you can specify as inputs when simulating data:
- `glitch_cfg_input_fn`: Glitch configuration file name (excluding file extensions). This is the file name of the yml file where you specify glitch parameters.
- `gw_cfg_input_fn`: Gravitational wave configuration file name (excluding file extensions). This is the file name of the yml file where you specify gravitational wave parameters.
- `pipe_cfg_input_fn`: Pipeline configuration file name (excluding file extensions). This is the file name of the yml file where you specify parameters specific to how LISA operates.
- `orbits_input_fn`: Orbit file name (excluding file extensions). This is an orbits file generated via `lisaorbits`. Generation code can be found in `make_orbits.py`. [`default="orbits"`]
- `pipe_output_fn`: Pipeline output file name (excluding file extensions). This will be the name of a .txt file containing general pipeline information. [`default="default_anomaly_output"`]
- `anomaly_output_fn`: Anomaly output file name (excluding file extensions). This will be the name of a .txt file containing anomaly shape, injection location (gw for gravitational waves), injection time, amplitude (meaning specified by user), duration (in s). [`default="default_anomaly_output"`]
- `simulation_output_fn`: Simulation output file name (excluding file extensions). This will be the name of a .h5 file containing simulation data (interferometer data, delays, etc.). [`default="default_simulation_output"`]
- `tdi_output_fn`: TDI output file name (excluding file extensions). This will be the name of a .h5 file containing data for $X_2$, $Y_2$, and $Z_2$. [`default="default_tdi_output"`]
- `disable_noise`: Boolean for whether or not to simulate LISA with noise. [`default=False`]

## Accessing Output Data
### Simulation and TDI Data (.h5 files)

Simulation data outputs can be found in `~/dist/lisa_data/simulation_data/`. Interferometer data can be read using the following code template:
~~~python
...
with h5py.File(<file_path>, "r") as sim_file:
    interferometer_data = sim_file[<interferometer_type> + "_carrier_fluctuations"][<mosa>],
...
~~~
Other simulation data can be accessed using a similar method and can be intuited by the user.

Conversly, TDI data ouptuts can be found in `~/dist/lisa_data/tdi_data/` and read in code using a similar code template:
~~~python 
...
with h5py.File(<file_path>, "r") as tdi_file:
    tdi_data = tdi_file[<tdi_channel>]
...
~~~

### Anomaly and Pipeline Information (.txt files)

To open anomlay and pipeline information (or really any .txt file for that matter), we use `numpy.genfromtxt`. The general template for accessing txt data outputted by `WilliamLISA` using this function is given below:
~~~python 
...
txt_data = np.genfromtxt(<txt_data_path>, dtype=<data_type>)

data = txt_data[<num_skipped_rows>:, <column_index>]
...
~~~
For further clarification, here is a code block that uses this template to extract data from an anomaly information file as a list of anomalies (where each anomaly is a dictionary with its respective information):
~~~python 
import numpy as np

anomaly_data_str = np.genfromtxt(<anomaly_data_path>, dtype=str)
anomaly_data_float = np.genfromtxt(<anomaly_data_path>, dtype=float)

shapes = anomaly_data_str[0:, 0]
inj_points = anomaly_data_str[0:, 1]
t_injs = anomaly_data_float[0:, 2]
amps = anomaly_data_float[0:, 3]
durations = anomaly_data_float[0:, 4]

num_anomalies = len(shapes)

anomalies = []
for i in range(num_anomalies):
    anomalies.append(
        {
            "i": i,
            "shape": shapes[i],
            "inj_point": inj_points[i],
            "t_inj": t_injs[i],
            "amp": amps[i],
            "duration": durations[i],
        }
    )
~~~

## Configuration File Structures
### Pipeline
Pipeline configuration files are .yml files and should be placed in `~/dist/pipe/pipe_config/`. These files should contain meta-information regarding LISA simulations. Below is the template for writing pipeline configuration files:
~~~yml 
dt: <float> s
duration: <float> s
~~~

### Individually Specified Glitch and Gravitational Waves
There are two types of configuration files for specifying the parameters of anomalies. This section is dedicated to the type where you can specify each anomaly's parameters individually. To do so, begin by creating .yml files in both `~/dist/glitch/glitch_config/` and `~/dist/gw/gw_config/`. To specify an anomaly, the following template for each anomaly is used in the configuration file of your choice:
~~~yml 
...
<glitch | gw>_<index>:
    inj_point: <str>
    t_inj: <float>
    ...
    <parameter_i>: <value>
    ...
...
~~~
Note that \<index> must start from 0. 

Below is an example of an individually specified configuration file for glitches:
~~~yml 
glitch_0:
  shape: "OneSidedDoubleExpGlitch"
  inj_point: "readout_tmi_carrier_12"
  t_inj: 10000
  t_rise: 100
  t_fall: 100
  level: 25000

glitch_1:
  shape: "StepGlitch"
  inj_point: "readout_isi_carrier_32"
  t_inj: 20000
  level: 1e4
~~~
Below is an example of an individually specified configuration file for gravitational waves:
~~~yml 
gw_0:
  shape: "GWFRED"
  t_inj: 30000
  t_rise: 50
  t_fall: 200
  level: 1e-13

gw_1:
  shape: "GWFRED"
  t_inj: 40000
  t_rise: 10
  t_fall: 25
  level: 1e-12
~~~

### Arbitrarily Specified Glitch and Gravitational Waves
The second type of configuration files for anomaly parameter definitions are those where you don't specify each anomaly and its paramaters directly, but rather you specify a range of parameters and a daily rate. Furthermore for glitches you should also specify the injection point.

Below is a template for an arbitrary glitch configuraiton file:
~~~yml 
shape: <str>
inj_points:
    ...
    - <str>
    ...
daily_rate: <int>
...
<parameter_i>_range: [<float>, <float>]
...
~~~
Below is a template for an arbitrary gravitational wave configuration file:
~~~yml 
shape: <str>
daily_rate: <int>
...
<parameter_i>_range: [<float>, <float>]
...
~~~

Like the individually specified anomaly configuration files, glitch configs should go in `~/dist/glitch/glitch_config/` and gravitational wave configs in `~/dist/gw/gw_config/`.

### Troubleshooting
- Currently mix-and-matching different anomaly shapes in arbitrarily specified configuration files is unsupported. So for instance, if you wanted an arbitrarily specified glitch config file with OneSidedDoubleExpGlitch glitches AND StepGlitch glitches, you would get an error or unexpected behaviour.
- You must have both non-empty glitch and non-empty gravitational wave config files in order to simulate data.
- Currently you are unable to mix-and-match configuration file types. For instance, if you wanted glitches to be individually specified and gravitational waves to be arbitrarily specified, you would get an error (likely).
- Due to a unexpected behaviour with LISA orbit files, each simulation starts at a time `t0=10368000` (defined on line 29 in `make_anomalies.py`). Thus we define injection times (`t_inj`) in configuration files as the time in seconds *after* `t0`. So a `t_inj` of 1 would correspond to an actual injection time of `t0` + 1 = 10368001.

# Modifying the Code

## Adding Glitch Shapes

The code is setup in such a way that it should be fairly ok to add new glitch shape support. This is done primarily through `lisaglitch`. There are two ways of doing this. The first is similar to adding new gravitational wave shapes via subclassing, so this section will go over the second which is adding shapes already implemented via `lisaglitch`, using `lisaglitch.StepGlitch` as an example. All the following work will be done in `~/src/simulate_lisa/make_anomalies.py`.

1. Import `lisaglitch.StepGlitch`:
    ~~~python
    from lisaglitch import StepGlitch
    ~~~
2. In `compute_anomalies_params` in the if-statement block that is called when a glitch is chosen to be made at random, add a new elif statement that appends a dictionary with all of the glitch's params to `glitches_params`:
    ~~~python
    ...
    elif glitch_cfg["shape"] == "StepGlitch":
        level_range = glitch_cfg["level_range"]

        glitches_params.append(
            {
                "shape": "StepGlitch",
                "inj_point": np.random.choice(glitch_cfg["inj_points"]),
                "level": np.random.uniform(float(level_range[0]), float(level_range[1])),
                "t_inj": t_inj,
            }
        )
    ...
    ~~~
3. In `compute_glitches`, add a new elif statement that creates the glitch object and appends it to `glitches`:
    ~~~python
    ...
    elif glitch_params["shape"] == "StepGlitch":
        glitches.append(
            StepGlitch(
                inj_point=glitch_params["inj_point"],
                t_inj=glitch_params["t_inj"],
                level=glitch_params["level"],
                t0=t0,
                size=size,
                dt=dt,
            )
        )
    ...
    ~~~
4. In `write` add the following elif statement for writing specific glitch information to the glitch txt data:
    ~~~python
        elif isinstance(glitch, StepGlitch):
            f.write(f"StepGlitch {glitch.t_inj} {glitch.level} {anomaly.duration}\n")
    ~~~
    Also add the following elif statement for writing specific glitch information to the anomnaly txt data:
    ~~~python
        elif isinstance(anomaly, StepGlitch):
            f.write(f"StepGlitch {anomaly.inj_point} {anomaly.t_inj} {anomaly.level} {anomaly.duration}\n")
    ~~~
Now as long as you follow the correct formatting for the configuration files as described in *Configuration File Structures*, you should be all good to go!

## Adding Gravitational Wave Shapes

Again, the code is setup in such a way that it should be fairly ok to add new gravitational wave shape support. This is done primarily through `lisagwresponse`. To do so will be almost identical to that for adding new glitch shapes, except instead of simply importing a class for the shape, we'll have to make our own. As an example, we'll add the FRED (Fast Rising Exponential Decay) gravitational wave shape. All the following work will be done in `~/src/simulate_lisa/make_anomalies.py`.

1. Create a new class for the gravitational wave shape by subclassing off of the abstract class `lisagwresponse.ResponseFromStrain`:
    ~~~python
    class GWFRED(ResponseFromStrain):
    """Represents a one-sided double-exponential gw signal

    Args:
        t_rise: Rising timescale
        t_fall: Falling timescale
        level: amplitude
    """
    def __init__(
        self,
        t_rise: float,
        t_fall: float,
        level: float,
        t_inj: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.t_rise = float(t_rise)
        self.t_fall = float(t_fall)
        self.level = float(level)
        self.t_inj = float(t_inj)

    def compute_hcross(self, t):
        # stub

    def compute_hplus(self, t):
        # stub
    ~~~

2. Implement the two methods `compute_hcross` and `compute_hplus` to apply the FRED mathematical shape and return the array `t` with FRED applied (only `compute_hcross` implementation is shown below):
    ~~~python
    ...
    def compute_hcross(self, t):
        """Computes the FRED response model.

        Args:
            t (array-like): Times to compute GW strain for.

        Returns:
            Computed FRED model (array-like)
        """
        offset = 20
        delta_t = t - self.t_inj + offset + (8.5 / 86400) * (self.t_inj - t0)

        if self.t_rise != self.t_fall:
            exp_terms = np.exp(-delta_t / self.t_rise) - np.exp(-delta_t / self.t_fall)
            signal = self.level * exp_terms / (self.t_rise - self.t_fall)
        else:
            signal = self.level * delta_t * np.exp(-delta_t / self.t_fall) / self.t_fall**2

        return np.where(delta_t >= 0, signal, 0)
    ...
    ~~~
3. In `compute_anomalies_params` in the if-statement block that is called when a gravitational wave is chosen to be made at random, add a new elif statement that appends a dictionary with all of the gravitational wave's params to `gws_params`:
    ~~~python
    ...
    elif gw_cfg["shape"] == "GWFRED":
        t_fall_range = gw_cfg["t_fall_range"]
        amp_range = gw_cfg["amp_range"]

        amp = np.random.uniform(float(amp_range[0]), float(amp_range[1]))
        t_fall = np.random.randint(t_fall_range[0], t_fall_range[1])
        level = amp * t_fall

        gws_params.append(
            {
                "shape": "GWFRED",
                "t_rise": t_fall,
                "t_fall": t_fall,
                "level": level,
                "t_inj": t_inj,
            }
        )
    ...
    ~~~
4. In `compute_gws`, add a new elif statement that creates the glitch object and appends it to `glitches`:
    ~~~python
    ...
    elif gw_params["shape"] == "GWFRED":
        gws.append(
            GWFRED(
                t_inj=gw_params["t_inj"],
                t_rise=gw_params["t_fall"],
                t_fall=gw_params["t_fall"],
                level=gw_params["level"],
                gw_beta=gw_beta,
                gw_lambda=gw_lambda,
                orbits=PATH_orbits_data + orbits_input_fn + ".h5",
                dt=dt,
                size=size,
                t0=t0,
            )
        )
    ...
    ~~~
5. In `write` add the following elif statement for writing specific gravitational wave information to the gravitational wave txt data:
    ~~~python
    elif isinstance(gw, GWFRED):
        f.write(f"GWFRED {gw.t_inj} {gw.amp} {gw.t_rise} {gw.t_fall} {gw.duration}\n")
    ~~~
    Also add the following elif statement for writing specific gravitational wave information to the anomnaly txt data:
    ~~~python
    elif isinstance(anomaly, GWFRED):
        f.write(f"GWBurst gw {anomaly.t_inj} {gw.amp} {gw.duration}\n")
    ~~~
If you repeat and adapt the above procedure for your given gravitational wave shape you should be able to start adding them to your simulations!