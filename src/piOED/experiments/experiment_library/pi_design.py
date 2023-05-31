import numpy as np
from copy import deepcopy

from external_packages.oed.src.piOED.experiments.interfaces.design_of_experiment import (
    Experiment,
)
from external_packages.oed.src.piOED.minimizer.interfaces.minimizer import Minimizer
from external_packages.oed.src.piOED.statistical_models.interfaces.statistical_model import StatisticalModel
from scipy.optimize import LinearConstraint, NonlinearConstraint


class PiDesign(Experiment):
    """parameter-individual experiment implemented within the experiment interface

    This experiment is calculated by minimizing a diagonal entry of the CRLB by changing experimental experiment.
    """
    def __init__(
            self,
            number_designs: int,
            lower_bounds_design: np.ndarray,
            upper_bounds_design: np.ndarray,
            index: int,
            initial_theta: np.ndarray,
            statistical_model: StatisticalModel,
            minimizer: Minimizer,
            constraints: {LinearConstraint, NonlinearConstraint} = (),
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
        index : int
            Index, i.e. diagonal entry, which should be minimized. Starts at zero.

        initial_theta : np.ndarray
            Parameter theta of the statistical models on which the Fisher information matrix is evaluated

        statistical_model : StatisticalModel
            Underlying statistical models implemented within the StatisticalModel interface

        minimizer : Minimizer
            Minimizer used to maximize the Fisher information matrix

        constraints : {LinearConstraint, NonlinearConstraint}
            Constraints used within the minimization

        setting: dict
            Setting of the experiment containing information about the experimental framework conditions
            e.g. {"number_experiments": 10, "time": np.array([0,7,14,21,...])}

        previous_experiment : Experiment
            Joint previously conducted experiment used within the maximization
            of the determinant of the Fisher information matrix
        """
        print(f"Calculating the {self.name}...")

        if previous_experiment is None:

            if setting is None:
                setting = {}
            self._design = [minimizer(
                                function=statistical_model.optimize_cramer_rao_lower_bound,
                                # TODO: check if settings are forwarded correctly
                                fcn_args=(initial_theta, number_designs, len(lower_bounds_design), index,
                                          setting.keys(), setting.values()),
                                lower_bounds=np.array(lower_bounds_design.tolist() * number_designs),
                                upper_bounds=np.array(upper_bounds_design.tolist() * number_designs),
                                constraints=constraints,
                                ).reshape(number_designs, len(lower_bounds_design))]
            self._setting = []
            self._setting.append(setting | {'number_experiments': number_designs})

            self._fim: list[np.ndarray] = []
            self._theta: list[np.ndarray] = []
        else:
            # If we want to consider an initial experiment within our calculation of the CRLB.
            self._design = previous_experiment.experiment
            if setting is None:
                setting = {}
            self._design.append(
                    minimizer(
                        function=statistical_model.optimize_cramer_rao_lower_bound,
                        # TODO: check if settings are forwarded correctly
                        fcn_args=(initial_theta, previous_experiment.fim, number_designs, len(lower_bounds_design), index,
                                  setting.keys(), setting.values()),
                        lower_bounds=np.array(lower_bounds_design.tolist() * number_designs),
                        upper_bounds=np.array(upper_bounds_design.tolist() * number_designs),
                        constraints=constraints,
                    ).reshape(number_designs, len(lower_bounds_design)),
                )

            self._setting = deepcopy(previous_experiment.setting)
            self._setting.append(setting | {'number_experiments': number_designs})
            self._fim = deepcopy(previous_experiment.fim)
            self._theta = deepcopy(previous_experiment.theta)

        print("finished!\n")

    @property
    def name(self) -> str:
        return "pi"

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
