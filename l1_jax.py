import jax
import jax.numpy as jnp


class LassoSolver:
    """Solves min (1/2n)||y - X@theta||^2 + lambda||theta||_1"""
    
    def __init__(self, method: str, lambda_: float, tol: float = 1e-4, max_iter: int = 1000, seed: int = 0):
        self.lambda_ = lambda_
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        if method == "cd":
            self.fit = self._fit_cd
        elif method == "gd":
            self.fit = self._fit_gd
        elif method == "jit_gd":
            self.fit = self._fit_gd_fully_jit
        else:
            raise NotImplementedError("No such method: {}".format(method))

    
    @staticmethod
    def _soft_threshold(theta: jnp.ndarray, threshold: float) -> jnp.ndarray:
        return jnp.sign(theta) * jnp.maximum(jnp.abs(theta) - threshold, 0)
    
    @staticmethod
    def _hard_threshold(theta: jnp.ndarray, threshold: float) -> jnp.ndarray:
        return jnp.where(jnp.abs(theta) > threshold, theta, 0.0)
    
    @staticmethod
    def _compute_lipschitz_constants(X: jnp.ndarray) -> jnp.ndarray:
        """Compute per-coordinate Lipschitz constants: L_j = ||X_j||^2 / n"""
        n_samples = X.shape[0]
        return jnp.sum(X ** 2, axis=0) / n_samples
    
    def _create_loss_fn(self, X: jnp.ndarray, y: jnp.ndarray, n_samples: int, include_l1: bool):
        if include_l1:
            def loss(theta):
                residual = y - X @ theta
                mse = 0.5 * jnp.dot(residual, residual) / n_samples
                l1 = jnp.linalg.norm(theta, 1)
                return mse + self.lambda_ * l1
        else:
            def loss(theta):
                residual = y - X @ theta
                mse = 0.5 * jnp.dot(residual, residual) / n_samples
                return mse
        return loss
    
    def _make_proximal_step(self, loss_fn, step_sizes: jnp.ndarray, thresholds: jnp.ndarray):
        def step(theta):
            grad = jax.grad(loss_fn)(theta)
            theta_new = theta - step_sizes * grad
            return self._soft_threshold(theta_new, thresholds)
        return step
    
    def _make_standard_step(self, loss_fn, step_sizes: jnp.ndarray):
        def step(theta):
            grad = jax.grad(loss_fn)(theta)
            return theta - step_sizes * grad
        return step
    
    def _fit_cd(self, X: jnp.ndarray, y: jnp.ndarray, lr: float = 0.01, exact: bool = False):
        """Coordinate descent with soft-thresholding"""
        n_samples, n_features = X.shape
        self.coef_ = jnp.zeros(n_features)
        
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features):
                residual = y - X @ self.coef_ + X[:, j] * self.coef_[j]
                rho = jnp.dot(X[:, j], residual) / n_samples
                z = jnp.dot(X[:, j], X[:, j]) / n_samples
                
                if z != 0:
                    self.coef_ = self.coef_.at[j].set(
                        jnp.sign(rho) * jnp.maximum(jnp.abs(rho) - self.lambda_, 0) / z
                    )
            
            if jnp.max(jnp.abs(self.coef_ - coef_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter
        
        return self
        
    def _fit_gd(self, X: jnp.ndarray, y: jnp.ndarray, lr: float = 0.01, exact: bool = False):
        """Gradient descent with partial JIT compilation.
        
        exact: If True, uses proximal GD with soft-thresholding for exact sparsity.
               If False (default), uses subgradient descent on full Lasso loss.
        """
        n_samples, n_features = X.shape
        
        lipschitz_constants = self._compute_lipschitz_constants(X)
        step_sizes = lr / (lipschitz_constants + 1e-8)
        loss = self._create_loss_fn(X, y, n_samples, include_l1=not exact)
        
        if exact:
            gd_step = jax.jit(self._make_proximal_step(loss, step_sizes, step_sizes * self.lambda_))
        else:
            gd_step = jax.jit(self._make_standard_step(loss, step_sizes))
        
        theta = jnp.zeros(n_features)
        for iteration in range(self.max_iter):
            theta_old = theta
            theta = gd_step(theta)
            
            if jnp.max(jnp.abs(theta - theta_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter
        
        if not exact:
            theta = self._hard_threshold(theta, jnp.max(step_sizes) * self.lambda_)
        
        self.coef_ = theta
        return self
    
    def _fit_gd_fully_jit(self, X: jnp.ndarray, y: jnp.ndarray, lr: float = 0.01, exact: bool = False):
        """Fully JIT-compiled GD with fori_loop. Runs for exactly max_iter (no early stopping)."""
        n_samples, n_features = X.shape
        
        lipschitz_constants = self._compute_lipschitz_constants(X)
        step_sizes = lr / (lipschitz_constants + 1e-8)
        loss = self._create_loss_fn(X, y, n_samples, include_l1=not exact)
        
        if exact:
            thresholds = step_sizes * self.lambda_
            step_fn = self._make_proximal_step(loss, step_sizes, thresholds)
        else:
            step_fn = self._make_standard_step(loss, step_sizes)
        
        def body_fun(i, theta):
            return step_fn(theta)
        
        @jax.jit
        def train(theta_init, num_iters):
            return jax.lax.fori_loop(0, num_iters, lambda i, theta: step_fn(theta), theta_init)
        
        theta = train(jnp.zeros(n_features), self.max_iter)
        
        if not exact:
            theta = self._hard_threshold(theta, jnp.max(step_sizes) * self.lambda_)
        
        self.coef_ = theta
        self.n_iter_ = self.max_iter
        return self
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        return X @ self.coef_
