# JAX Lasso Regression Implementation

An educational repository for learning JAX basics and implementing classic L1-regularized least squares (Lasso) regression algorithms with comprehensive profiling and experiment logging.

## Overview

This project implements multiple Lasso regression solvers using JAX, comparing them against scikit-learn baselines. The main goal is to demonstrate JAX's capabilities for numerical optimization while recovering classic algorithms like coordinate descent and gradient descent with various acceleration techniques.

- **scikit-learn remains fastest overall** due to highly optimized C++ implementations
- **JIT-compiled methods are quite competitive** and often faster per iteration

## Implementations

### Custom JAX Solvers

1. **Coordinate Descent (`cd`)** - Classic soft-thresholding updates, one coefficient at a time
2. **Gradient Descent (`gd`)** - Partial JIT compilation with adaptive per-coordinate step sizes
3. **Fully JIT Gradient Descent (`jit_gd`)** - Complete JIT compilation using `jax.lax.fori_loop`

Each solver supports two modes:
- **`exact=True`**: Proximal gradient descent with soft-thresholding for exact sparsity
- **`exact=False`**: Standard subgradient descent on full Lasso loss with hard-thresholding

### Baselines

- **scikit-learn Lasso**
- **scikit-learn LassoLars**

## Usage

```python
# Generate synthetic data
generator = LassoDataGenerator(d=40, sparsity=0.3, noise_std=0.1, seed=42)
X_train, y_train = generator.generate_dataset(200)
X_test, y_test = generator.generate_dataset(100)

# Train with different solvers
solver = LassoSolver(method='jit_gd', lambda_=0.05, max_iter=1000)
solver.fit(X_train, y_train, lr=0.01, exact=True)

# Evaluate performance
y_pred = solver.predict(X_test)
mse = jnp.mean((y_test - y_pred) ** 2)
```

## Running Experiments

```bash
# Run comprehensive comparison
python test_l1_jax.py

# View experiment summary
python db_summary.py
```

## Educational Value

This repository demonstrates:
- JAX's automatic differentiation capabilities
- JIT compilation strategies for numerical optimization
- Classic optimization algorithms (coordinate descent, gradient descent, ISTA)
- Performance profiling and experiment management

This README and most experiment handling code written with Cursor. All errors my own.