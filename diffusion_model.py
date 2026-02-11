import torch
import math


class DiffusionModel(torch.nn.Module):

    def __init__(self, model: torch.nn, step_size: int, noise_schedule, mu_function, sigma_function, lr:float = 0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model = model
        self._dt = step_size
        self._optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        self._loss_function = torch.nn.MSELoss()
        self._beta_function = noise_schedule
        self._mu_function = mu_function
        self._sigma_function = sigma_function

        self._f = lambda x, t: -0.5 * self._beta_function(t) * x
        self._g = lambda t: torch.sqrt(self._beta_function(t))

    # Training Functions
    def _update_weights(self, samples: torch.tensor):
        self._model.train()

        # Arbitrarily diffuse to a specific random timestep
        # Add one more iteration of noise to the samples, saving this noise
        noised_samples, noise, times = self.__diffuse_to_time(samples)

        # Feed noised sample through model
        estimated_noise = self._model(noised_samples, times)

        # Compute MSE of noise
        loss = self._loss(estimated_noise, noise)

        # zero grad
        self._optimiser.zero_grad()

        # Backpropagate
        loss.backward()
        self._optimiser.step()

    def __diffuse_to_time(self, samples: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        # Generate an array of times in the diffusion interval
        times = (torch.rand(samples.shape[0])).reshape(samples.shape[0], *([1] * (samples.ndim - 1)))

        # x_t = mu(t)x_0 + sigma(t) epsilon
        epsilon = torch.randn_like(samples)

        diffused_samples = (self._mu_function(times) * samples +
                            self._sigma_function(times) * epsilon)

        return diffused_samples, epsilon, times

    def train_model(self, data: torch.utils.data.DataLoader, epochs: int):
        for e in range(epochs):
            for samples in data:
                self._update_weights(samples)

    # Evaluation functions
    def forward(self, X: torch.tensor):
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
        return self._model(X, t) / self._sigma_function(t)
