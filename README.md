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
with h5py.File(<file_path>, "r") as sim_file:
    interferometer_data = sim_file[<interferometer_type> + "_carrier_fluctuations"][<mosa>],
~~~
Other simulation data can be accessed using a similar method and can be intuited by the user.

Conversly, TDI data ouptuts can be found in `~/dist/lisa_data/tdi_data/` and read in code using a similar code template:
~~~python 
with h5py.File(<file_path>, "r") as tdi_file:
    tdi_data = tdi_file[<tdi_channel>]
~~~

### Anomaly and Pipeline Information (.txt files)

To open anomlay and pipeline information (or really any .txt file for that matter), we use `numpy.genfromtxt`. The general template for accessing txt data outputted by `WilliamLISA` using this function is given below:
~~~python 
txt_data = np.genfromtxt(<txt_data_path>, dtype=<data_type>)

data = txt_data[<num_skipped_rows>:, <column_index>]
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
<glitch | gw>_<index>:
    inj_point: <str>
    t_inj: <float>
    ...
    <parameter_i>: <value>
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

The code is setup in such a way that it should be fairly ok to add new glitch shape support. This is done primarily through `lisaglitch`. As an example, let's walk through how to add the ReducedOneSidedDoubleExp glitch shape already implemented.

1. In `~/src/simulate_lisa/glitch_shapes.py`, create a new subclass from the abstract class `lisaglitch.Glitch` defining the methods `comptue_duration` and `compute_signal`. Also add any parameters your model may need as well as defining a `duration` property that is equal to whatever you implement `compute_duration` to be. Furthermore, I'd recomend mathematically deriving elements in the model directly contribute to amplitude so you can use amplitude as a parameter, then define other parameters that actually go into the computation of the model using the amplitude you inputted. This allows a decoupling between duration and amplitude of the glitch.
    ~~~python
    class ReducedOneSidedDoubleExpGlitch(Glitch):
        """Represents a one-sided double exponential glitch in the case where t_rise=t_fall

        Args:
            t_rise: Rising timescale
            t_fall: Falling timescale
            amp: relative amplitude
        """

        def __init__(
            self,
            t_fall: float,
            amp: float,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)

            self.t_fall = float(t_fall)
            self.amp = float(amp)
            self.level = self.amp * self.t_fall
            self.duration = self.compute_duration()

        def compute_duration(self) -> float:
            # stub

        def compute_signal(self, t) -> np.ndarray:
            # stub
    ~~~
2. Implement the `compute_duration` function. This should return an approximation for the duration of the glitch. The below implementation returns the time at which a `ReducedOneSidedDoubleExpGlitch` glitch reaches $\approx 3\%$ of its maximum amplitude.
    ~~~python
    def compute_duration(self) -> float:
        """compute an approximate duration for the glitch"""
        roots = lambda t: self.compute_signal(t) - self.amp / 30

        guess = self.t_fall + 50

        return float(fsolve(roots, guess)[0])
    ~~~
3. Implement the `compute_signal` function takes in an array of times and returns the mathematical model for your glitch shape applied to those times.
    ~~~python
    def compute_signal(self, t) -> np.ndarray:
        """Computes the FRED response model.

        Args:
            t (array-like): Times to compute glitch model for.

        Returns:
            Computed FRED model (array-like)
        """
        delta_t = t - self.t_inj

        signal = self.level * delta_t * np.exp(-delta_t / self.t_fall) / self.t_fall**2

        return np.where(delta_t >= 0, signal, 0)
    ~~~
4. Now all the remaining steps will be done in `~/src/simulate_lisa/make_anomalies.py`. In `compute_anomalies_params` in the if-statement block that is called when a glitch is chosen to be made at random, add a new elif statement that appends a dictionary with all of the glitch's params to `glitches_params`.
    ~~~python
    ...
    elif glitch_cfg["shape"] == "ReducedOneSidedDoubleExpGlitch":
        t_fall_range = glitch_cfg["t_fall_range"]
        amp_range = glitch_cfg["amp_range"]

        glitches_params.append(
            {
                "shape": "ReducedOneSidedDoubleExpGlitch",
                "inj_point": np.random.choice(glitch_cfg["inj_points"]),
                "t_fall": np.random.randint(t_fall_range[0], t_fall_range[1]),
                "amp": np.random.uniform(float(amp_range[0]), float(amp_range[1])),
                "t_inj": t_inj,
            }
        )
    ...
    ~~~
5. In `compute_glitches`, add a new elif statement that creates the glitch object and appends it to `glitches`.
    ~~~python
    ...
    elif glitch_params["shape"] == "ReducedOneSidedDoubleExpGlitch":
        glitches.append(
            ReducedOneSidedDoubleExpGlitch(
                inj_point=glitch_params["inj_point"],
                t_inj=glitch_params["t_inj"],
                t_fall=glitch_params["t_fall"],
                amp=glitch_params["amp"],
                t0=t0,
                size=size,
                dt=dt,
            )
        )
    ...
    ~~~
6. In `write` add the following elif statement for writing specific glitch information to the glitch txt data.
    ~~~python
    ...
    elif isinstance(glitch, ReducedOneSidedDoubleExpGlitch):
        f.write(f"ReducedOneSidedDoubleExpGlitch {glitch.t_inj} {glitch.amp} {glitch.t_fall} {glitch.duration}\n")
    ...
    ~~~
    Also add the following elif statement for writing specific glitch information to the anomnaly txt data:
    ~~~python
    ...
    elif isinstance(anomaly, ReducedOneSidedDoubleExpGlitch):
        f.write(f"ReducedOneSidedDoubleExpGlitch {anomaly.inj_point} {anomaly.t_inj} {anomaly.amp} {anomaly.duration}\n")
    ...
    ~~~
Now as long as you follow the correct formatting for the configuration files as described in *Configuration File Structures*, you should be all good to go!

## Adding Gravitational Wave Shapes

Again, the code is setup in such a way that it should be fairly ok to add new gravitational wave shape support. This is done primarily through `lisagwresponse`. To do so will be almost identical to that for adding new glitch shapes. As an example, we'll add the ReducedOneSidedDoubleExp gravitational wave shape just as we had done for glitches.

1. In `~/src/simulate_lisa/gw_shapes.py`, create a new subclass from the abstract class `lisagwresponse.ResponseFromStrain` defining the methods `comptue_duration`, `compute_hcross` and `compute_hplus`. Also add any parameters your model may need as well as defining a `duration` property that is equal to whatever you implement `compute_duration` to be. Furthermore, I'd recomend mathematically deriving elements in the model directly contribute to amplitude so you can use amplitude as a parameter, then define other parameters that actually go into the computation of the model using the amplitude you inputted. This allows a decoupling between duration and amplitude of the gravitational wave.
    ~~~python
    class ReducedOneSidedDoubleExpGW(ResponseFromStrain):
        """Represents a one-sided double-exponential gw in the case where t_rise=t_fall

        Args:
            t_rise: Rising timescale
            t_fall: Falling timescale
            amp: relative amplitude scale
        """
        def __init__(
            self,
            t_fall: float,
            amp: float,
            t_inj: float,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)

            self.t_fall = float(t_fall)
            self.amp = float(amp)
            self.level = self.amp * self.t_fall
            self.t_inj = float(t_inj)
            self.duration = self.compute_duration()

        def compute_duration(self) -> float:
            # stub

        def compute_hcross(self, t) -> np.ndarray:
            # stub

        def compute_hplus(self, t) -> np.ndarray:
            # stub
    ~~~
2. Implement the `compute_duration` function. This should return an approximation for the duration of the gravitational wave. The below implementation returns the time at which a `ReducedOneSidedDoubleExpGW` gravitational wave reaches $\approx 3\%$ of its maximum amplitude.
    ~~~python
    def compute_duration(self) -> float:
        """compute an approximate duration for the gw"""
        roots = lambda t: self.compute_signal(t) - self.amp / 30

        guess = self.t_fall + 50

        return float(fsolve(roots, guess)[0])
    ~~~
3. Implement the two methods `compute_hcross` and `compute_hplus` to apply the mathematical shape for your gravitational wave and return the array `t` with shape applied (only `compute_hcross` implementation is shown below).
    ~~~python
    def compute_signal(self, t) -> np.ndarray:
        """Computes the one-sided double exponential model in the case where t_rise=t_fall.

        Args:
            t (array-like): Times to compute GW model for.

        Returns:
            Computed model (array-like)
        """
        offset = 405
        delta_t = t - self.t_inj + offset

        signal = self.level * delta_t * np.exp(-delta_t / self.t_fall) / self.t_fall**2

        return np.where(delta_t >= 0, signal, 0)
    ~~~
4. Now all the remaining steps will be done in `~/src/simulate_lisa/make_anomalies.py`. In `compute_anomalies_params` in the if-statement block that is called when a gravitational wave is chosen to be made at random, add a new elif statement that appends a dictionary with all of the gravitational wave's params to `gws_params`.
    ~~~python
    ...
    elif gw_cfg["shape"] == "ReducedOneSidedDoubleExpGW":
        t_fall_range = gw_cfg["t_fall_range"]
        amp_range = gw_cfg["amp_range"]

        gws_params.append(
            {
                "shape": "ReducedOneSidedDoubleExpGW",
                "t_fall": np.random.randint(t_fall_range[0], t_fall_range[1]),
                "amp": np.random.uniform(float(amp_range[0]), float(amp_range[1])),
                "t_inj": t_inj,
            }
        )
    ...
    ~~~
5. In `compute_gws`, add a new elif statement that creates the glitch object and appends it to `glitches`.
    ~~~python
    ...
    elif gw_params["shape"] == "ReducedOneSidedDoubleExpGW":
        gws.append(
            ReducedOneSidedDoubleExpGW(
                t_inj=gw_params["t_inj"],
                t_fall=gw_params["t_fall"],
                amp=gw_params["amp"],
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
6. In `write` add the following elif statement for writing specific gravitational wave information to the gravitational wave txt data.
    ~~~python
    ...
    if isinstance(gw, ReducedOneSidedDoubleExpGW):
        f.write(f"ReducedOneSidedDoubleExpGW {gw.t_inj} {gw.amp} {gw.t_fall} {gw.duration}\n")
    ...
    ~~~
    Also add the following elif statement for writing specific gravitational wave information to the anomnaly txt data:
    ~~~python
    ...
    elif isinstance(anomaly, ReducedOneSidedDoubleExpGW):
        f.write(f"ReducedOneSidedDoubleExpGW gw {anomaly.t_inj} {anomaly.amp} {anomaly.duration}\n")
    ...
    ~~~
If you repeat and adapt the above procedure for your given gravitational wave shape you should be able to start adding them to your simulations!