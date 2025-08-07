import numpy as np
from scipy.optimize import fsolve
from lisagwresponse import ResponseFromStrain


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
        """compute an approximate duration for the gw"""
        roots = lambda t: self.compute_signal(t) - self.amp / 30

        guess = self.t_fall + 50

        return float(fsolve(roots, guess)[0])

    def compute_hcross(self, t) -> np.ndarray:
        return self.compute_signal(t)

    def compute_hplus(self, t) -> np.ndarray:
        return self.compute_signal(t)

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
