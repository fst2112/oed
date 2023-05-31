from abc import ABC, abstractmethod

import numpy as np


class Experiment(ABC):
    """Interface for an experiment

    We refer to an experiment as vector of experimental experiment.
    We store an experiment as a numpy array with each entry representing an experimental design.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the experiment
        Returns
        -------
        str
            the name of the experiment
        """
        pass

    @property
    @abstractmethod
    def experiment(self) -> list[np.ndarray]:
        """Experiment consisting of experimental designs

        We refer to an experiment as vector of experimental experiment.
        We store an experiment as a numpy array with each entry representing an experimental design.

        Returns
        -------
        np.ndarray
            experiment
        """
        pass

    @property
    @abstractmethod
    def setting(self) -> list[dict]:
        """Setting of the experiment containing information about the experimental framework conditions
        e.g. [{"number_experiments": 10, "time": np.array([0,7,14,21,...])}]

        Returns
        -------
        dict
            setting
        """
        pass

    @property
    @abstractmethod
    def fim(self) -> list[np.ndarray]:
        """Fisher information matrix of the experiment per iteration. This is a list of numpy arrays.
        e.g. [np.array([]), np.array([]), ...

        Returns
        -------
        list[np.ndarray]
            FIM
        """
        pass

    @property
    @abstractmethod
    def theta(self) -> list[np.ndarray]:
        """Parameter vector of the experiment per iteration. This is a list of numpy arrays.
        e.g. [np.array([]), np.array([]), ...
        """
        pass
