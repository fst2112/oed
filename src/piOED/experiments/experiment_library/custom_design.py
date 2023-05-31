import numpy as np
from copy import deepcopy

from external_packages.oed.src.piOED.experiments.interfaces.design_of_experiment import (
    Experiment,
)


class Custom(Experiment):
    """Custom design implemented within the experiment interface

    Enables to load custom designs to the framework
    """
    def __init__(
        self,
        design: np.ndarray,
        setting: dict = None,
        previous_experiment: Experiment = None,
    ):
        """

        Parameters
        ----------
        design: np.ndarray
            Custom design
        setting: dict
            Setting of the experiment containing information about the experimental framework conditions
            e.g. {"number_experiments": 10, "time": np.array([0,7,14,21,...])}
        previous_experiment: Experiment
            Previous experiment to be considered in the calculation of the CRLB
        """

        if previous_experiment is None:
            self._design = [design]
            self._setting = []
            if setting is None:
                setting = {}
            self._setting.append(setting | {'number_experiments': design.shape[0]})
            self._fim: list[np.ndarray] = []
            self._theta: list[np.ndarray] = []
        else:
            # If we want to consider an initial experiment within our calculation of the CRLB.
            self._design = previous_experiment.experiment.append(design)
            if setting is None:
                setting = {}
            self._setting = deepcopy(previous_experiment.setting)
            self._setting.append(setting | {'number_experiments': design.shape[0]})
            self._fim = deepcopy(previous_experiment.fim)
            self._theta = deepcopy(previous_experiment.theta)

    @property
    def name(self) -> str:
        return "Custom"

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
