import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional


class LassoDataGenerator:
    """Generates synthetic data: y = X @ theta_0 + eps"""
    
    def __init__(
        self,
        d: int,
        theta_0: Optional[jnp.ndarray] = None,
        mean: Optional[jnp.ndarray] = None,
        cov_diag: Optional[jnp.ndarray] = None,
        noise_std: float = 0.1,
        sparsity: float = 0.5,
        seed: int = 0
    ):
        self.d = d
        self.noise_std = noise_std
        self.key = random.PRNGKey(seed)
        
        self.theta_0 = self._generate_sparse_theta(d, sparsity) if theta_0 is None else jnp.array(theta_0)
        self.mean = jnp.zeros(d) if mean is None else jnp.array(mean)
        self.cov_diag = jnp.ones(d) if cov_diag is None else jnp.array(cov_diag)
        
        assert self.theta_0.shape == (d,)
        assert self.mean.shape == (d,)
        assert self.cov_diag.shape == (d,)
    
    def _generate_sparse_theta(self, d: int, sparsity: float) -> jnp.ndarray:
        assert 0 <= sparsity <= 1
        
        key_vals, key_mask = random.split(self.key, 2)
        self.key = random.split(self.key)[0]
        
        theta = random.normal(key_vals, shape=(d,))
        n_nonzero = max(1, int(d * sparsity))
        mask = jnp.zeros(d)
        indices = random.choice(key_mask, d, shape=(n_nonzero,), replace=False)
        mask = mask.at[indices].set(1.0)
        
        return theta * mask
    
    def generate_dataset(self, N: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (X, y) where X has shape (N, d) and y has shape (N,)"""
        key_X, key_eps = random.split(self.key, 2)
        self.key = random.split(self.key)[0]
        
        X_standard = random.normal(key_X, shape=(N, self.d))
        X = X_standard * jnp.sqrt(self.cov_diag) + self.mean
        eps = random.normal(key_eps, shape=(N,)) * self.noise_std
        y = X @ self.theta_0 + eps
        
        return X, y
    
    def get_true_theta(self) -> jnp.ndarray:
        return self.theta_0

