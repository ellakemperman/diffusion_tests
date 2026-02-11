import torch
import math
from typing import Callable


class DiffusionModel(torch.nn.Module):

    def __init__(
            self,
            model: torch.nn,
            n_steps: int,
            noise_schedule: Callable[[torch.tensor], torch.tensor],
            mu_function: Callable[[torch.tensor], torch.tensor],
            sigma_function: Callable[[torch.tensor], torch.tensor],
            lr:float = 0.001,
            *args, **kwargs
            ):
        """
        Constructs the diffusion model.

        :param model: Model that maps an object of shape :math:`(D_0, D_1, ... D_n)` with :math:`n \geq 1` to an object of the same shape
        :param n_steps: Number of diffusion steps to be taken, should be greater than 0
        :param noise_schedule: Noise schedule function, maps a float -> float deterministically for each element in the tensor.
        :param mu_function: Function derived from the noise schedule
        :param sigma_function: Function derived from the noise schedule
        :param lr: Learning rate
        """
        super().__init__(*args, **kwargs)


        self._dt = 1 / n_steps

        # Pytorch stuff
        self._model = model
        self._optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        self._loss_function = torch.nn.MSELoss()

        # All based on noise schedule beta
        self._beta_function = noise_schedule
        self._mu_function = mu_function
        self._sigma_function = sigma_function
        self._f = lambda x, t: -0.5 * self._beta_function(t) * x  # Drift
        self._g = lambda t: torch.sqrt(self._beta_function(t))  # Diffusion

    # Training Functions
    def _update_weights(self, X: torch.tensor) -> 'DiffusionModel':
        """
        Performs one diffusion training step. Performs training on X, by diffusing them to a random point of time in
        the interval [0, 1], saving this noise, sending the diffused signal through a model, which goal is to predict
        this noise, then training on MSELoss of this noise and the ground truth noise.

        :param X: A torch.tensor of shape (batch_size, D_0, ..., D_n), that matches the input dimensions of the model.
        :return: Itself
        """
        self._model.train()

        # Arbitrarily diffuse to a specific random timestep, save noise
        X_t, epsilon, t = self.__diffuse_to_time(X)

        # Feed noised sample through model
        estimated_noise = self._model(X_t, t)

        # Compute MSE of noise
        loss = self._loss(estimated_noise, epsilon)

        # zero grad
        self._optimiser.zero_grad()

        # Backpropagate
        loss.backward()
        self._optimiser.step()

        return self

    def __diffuse_to_time(self, X: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        r"""
        Diffuses X to a different random time in :math:`t \sim U([0, 1])`, where each sample in the batch gets assigned
        a different time t. Uses the closed form equation :math:`X_t = \mu(t)x_0 + \sigma(t) \epsilon` to obtain the
        noise :math:`epsilon` and the diffused sample :math:`X_t`.

        :param X: A torch.tensor of shape (batch_size, D_0, ..., D_n), that matches the input dimensions of the model.
        :return: A tuple of form :math:`(X_t, epsilon, t)`
        """
        # Generate an array of times in the diffusion interval
        t = (torch.rand(X.shape[0])).reshape(X.shape[0], *([1] * (X.ndim - 1)))

        # X_t = mu(t)x_0 + sigma(t) epsilon
        epsilon = torch.randn_like(X)

        X_t = (self._mu_function(t) * X +
               self._sigma_function(t) * epsilon)

        return X_t, epsilon, t

    def train_model(self, data: torch.utils.data.DataLoader, epochs: int):
        """
        Trains the diffusion model on a given dataset.

        :param data: The dataset to train on, dimensionality should match model dimensionality
        :param epochs: Number of epochs to train for
        :return: Itself
        """
        for e in range(epochs):
            for samples in data:
                self._update_weights(samples)
        return self

    # Evaluation functions
    def forward(self, X: torch.tensor):
        """
        Performs generation, implementing the backwards SDE by Song et al (2021)

        :param X: Latent vector input
        :return: :math:`X'`, the latent vector input transformed from a Gaussian distribution to the target distribution
        """
        self._model.eval()

        # Perform Euler integration
        X_prime = X
        for t_ in range(1, 0, -self._dt):
            t = torch.full((X.shape[0],), t_)
            dw = torch.randn_like(X) * math.sqrt(self._dt)

            # Backwards SDE (Song et al)
            X_prime = X_prime + (self._f(X_prime, t) - self._g(t) ** 2 * self._score_function(X_prime, t)) * self._dt + self._g(t) * dw

        return X_prime

    def _score_function(self, X: torch.tensor, t: torch.tensor) -> torch.tensor:
        """
        Gives the score of X at a time point t

        :param X: The input data of which the score has to be determined.
        :param t: The time at which the score is determined
        :return: :math:`\del_X \log(p_t(X))
        """
        return self._model(X, t) / self._sigma_function(t)
