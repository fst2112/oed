import numpy as np

from external_packages.oed.src.piOED.minimizer.interfaces.minimizer import Minimizer
from external_packages.oed.src.piOED.parametric_function_library.interfaces.parametric_function import (
    ParametricFunction,
)
from external_packages.oed.src.piOED.statistical_models.interfaces.statistical_model import StatisticalModel


class GaussianNoiseModel(StatisticalModel):
    """Implementation of the statistical model induced by a function with white Gaussian noise
    ...within the StatisticalModel interface

    We specify a function f and a variance standard deviation sigma. The statistical model at some experimental experiment x
    is then given by the normal distribution N(f(x),sigma^2).
    Accordingly, given an experiment x0 consisting of experimental experiment x_1,...,x_n, the corresponding
    statistical model is then given by the multivariate normal distribution with mean vector (f(x))_{x \in x0}
    and covariance matrix diagonal matrix with all diagonal entries equal to sigma**2.
    """

    def __init__(
            self,
            function: ParametricFunction,
            lower_bounds_theta: np.ndarray,
            upper_bounds_theta: np.ndarray,
            lower_bounds_x: np.ndarray,
            upper_bounds_x: np.ndarray,
            sigma: float = 1,
    ) -> None:
        """
        Parameters
        ----------
        function : ParametricFunction
            Parametric function parametrized by theta.
        sigma : float
            Standard deviation of the underlying white noise in each component (default is 1)
        """
        self._function = function
        self._var = sigma ** 2
        self._lower_bounds_theta = lower_bounds_theta
        self._upper_bounds_theta = upper_bounds_theta
        self._lower_bounds_x = lower_bounds_x
        self._upper_bounds_x = upper_bounds_x

    def __call__(self, x: np.ndarray, theta: np.ndarray, **kwargs) -> np.ndarray:
        return self._function(theta=theta, x=x, **kwargs)

    def random(self, x: np.ndarray, theta: np.ndarray, **kwargs) -> np.ndarray:
        return np.random.normal(
            loc=self._function(theta=theta, x=x, **kwargs), scale=np.sqrt(self._var)
        )

    def calculate_fisher_information(
            self, theta: np.ndarray, i: int, j: int, x0: np.ndarray, **kwargs
    ):
        return (1 / self._var * np.dot(self._function.partial_derivative(theta=theta, x=x0, parameter_index=i, **kwargs).flatten().T,
                                       self._function.partial_derivative(theta=theta, x=x0, parameter_index=j, **kwargs).flatten()))

    def calculate_fisher_information_matrix(
            self, x0: np.ndarray, theta: np.ndarray, **kwargs
    ) -> np.ndarray:
        k = len(theta)
        return np.array(
            [
                [
                    self.calculate_fisher_information(theta=theta, x0=x0, i=i, j=j, **kwargs)
                    for i in range(k)
                ]
                for j in range(k)
            ]
        )

    def calculate_log_likelihood(
            self, theta: np.ndarray, x0: np.ndarray, y: np.ndarray, *args
    ) -> np.ndarray:
        kwargs = dict(zip(args[0], args[1]))
        yhat = self._function(theta=theta, x=x0, **kwargs)
        return np.sum(np.square((y - yhat)))

    def calculate_likelihood(
            self, x0: np.ndarray, y: np.ndarray, theta: np.ndarray, **kwargs
    ) -> float:
        return np.exp(self.calculate_log_likelihood(x0=x0, y=y, theta=theta, **kwargs))

    def calculate_partial_derivative_log_likelihood(
            self, x0: np.ndarray, y: np.ndarray, theta: np.ndarray, parameter_index: int, **kwargs
    ) -> np.ndarray:
        return -np.sum(
            (y - np.array([self._function(theta=theta, x=x, **kwargs) for x in x0]))
            * np.array(
                [
                    self._function.partial_derivative(
                        theta=theta, x=x, parameter_index=parameter_index, **kwargs,
                    )
                    for x in x0
                ]
            )
        )

    def calculate_maximum_likelihood_estimation(
            self, x0: np.ndarray, y: np.ndarray, minimizer: Minimizer, **kwargs
    ) -> np.ndarray:
        args = None
        kwargs_keys = None

        if len(kwargs) > 0:
            args = kwargs.values()
            kwargs_keys = kwargs.keys()

        return minimizer(
            function=self.calculate_log_likelihood,
            fcn_args=(x0, y, kwargs_keys, args),
            lower_bounds=self.lower_bounds_theta,
            upper_bounds=self.upper_bounds_theta,
        )

    @property
    def lower_bounds_theta(self) -> np.ndarray:
        return self._lower_bounds_theta

    @property
    def upper_bounds_theta(self) -> np.ndarray:
        return self._upper_bounds_theta

    @property
    def lower_bounds_x(self) -> np.ndarray:
        return self._lower_bounds_x

    @property
    def upper_bounds_x(self) -> np.ndarray:
        return self._upper_bounds_x

    @property
    def name(self) -> str:
        return "Gaussian white noise model"
