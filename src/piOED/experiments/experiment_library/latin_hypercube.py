import numpy as np
from scipy.stats import qmc
from copy import deepcopy

from external_packages.oed.src.piOED.experiments.interfaces.design_of_experiment import Experiment


class LatinHypercube(Experiment):
    """Latin Hypercube design implemented within the experiment interface

    See https://en.wikipedia.org/wiki/Latin_hypercube_sampling.
    """
    def __init__(
        self,
        number_designs: int,
        lower_bounds_design: np.ndarray,
        upper_bounds_design: np.ndarray,
        setting: dict = None,
        previous_experiment: Experiment = None,
    ):
        """

        Parameters
        ----------
        number_designs : int
            The number of experimental experiment over which the maximization is taken

        lower_bounds_design : np.ndarray
            Lower bounds for an experimental experiment x
            with each entry representing the lower bound of the respective entry of x

        upper_bounds_design :  np.ndarray
            Lower bounds for an experimental experiment x
            with each entry representing the lower bound of the respective entry of x

        setting: dict
            Setting of the experiment containing information about the experimental framework conditions
            e.g. {"number_experiments": 10, "time": np.array([0,7,14,21,...])}

        previous_experiment: Experiment
            Previous experiment to be considered in the calculation of the CRLB
        """

        if previous_experiment is None:
            self._design = [qmc.scale(
                qmc.LatinHypercube(d=len(lower_bounds_design)).random(n=number_designs),
                lower_bounds_design,
                upper_bounds_design,
            )]
            self._setting = []
            if setting is None:
                setting = {}
            self._setting.append(setting | {'number_experiments': self._design[0].shape[0]})

            self._fim: list[np.ndarray] = []
            self._theta: list[np.ndarray] = []

        else:
            # If we want to consider an initial experiment within our calculation of the CRLB.
            self._design = previous_experiment.experiment
            self._design.append(qmc.scale(
                        qmc.LatinHypercube(d=len(lower_bounds_design)).random(n=number_designs),
                        lower_bounds_design,
                        upper_bounds_design,
                    ))

            if setting is None:
                setting = {}
            self._setting = deepcopy(previous_experiment.setting)
            self._setting.append(setting | {'number_experiments': self._design[0].shape[0]})
            self._fim = deepcopy(previous_experiment.fim)
            self._theta = deepcopy(previous_experiment.theta)

    @property
    def name(self) -> str:
        return "LH"

    @property
    def experiment(self) -> list[np.ndarray]:
        return self._design

    @property
    def setting(self) -> list[dict]:
        return self._setting

    @property
    def fim(self) -> list[np.ndarray]:
        return self._fim

    @property
    def theta(self) -> list[np.ndarray]:
        return self._theta