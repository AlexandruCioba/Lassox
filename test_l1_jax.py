import time
import numpy as np
import jax.numpy as jnp
from lasso_data_generator import LassoDataGenerator
from l1_jax import LassoSolver
from sklearn.linear_model import Lasso, LassoLars
from experiment_db import ExperimentDB


def evaluate_solver(solver, X_train, y_train, X_test, y_test, true_theta, fit_kwargs=None):
    if fit_kwargs is None:
        fit_kwargs = {}
    
    start_time = time.time()
    solver.fit(X_train, y_train, **fit_kwargs)
    elapsed_time = time.time() - start_time
    
    coef = solver.coef_
    n_iter = solver.n_iter_
    sparsity = int(jnp.count_nonzero(coef))
    
    y_train_pred = solver.predict(X_train)
    y_test_pred = solver.predict(X_test)
    
    train_mse = float(jnp.mean((y_train - y_train_pred) ** 2))
    test_mse = float(jnp.mean((y_test - y_test_pred) ** 2))
    param_error = float(jnp.linalg.norm(coef - true_theta))
    time_per_iter = elapsed_time / n_iter if n_iter > 0 else 0.0
    
    return {
        'coef': coef,
        'n_iter': n_iter,
        'sparsity': sparsity,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'param_error': param_error,
        'time': elapsed_time,
        'time_per_iter': time_per_iter
    }


def main():
    print("=" * 70)
    print("Lasso Regression Example")
    print("=" * 70)
    
    lambda_ = 0.05
    lr = 0.01
    max_iter = 1000
    tol = 1e-4
    
    d = 40
    sparsity = 0.3
    noise_std = 0.1
    N_train = 200
    N_test = 100
    seed = 42
    
    db = ExperimentDB("lasso_experiments.db")
    
    print(f"\nHyperparameters:")
    print(f"  lambda: {lambda_}")
    print(f"  learning rate: {lr}")
    print(f"  max_iter: {max_iter}")
    print(f"  tol: {tol}")
    
    generator = LassoDataGenerator(d=d, sparsity=sparsity, noise_std=noise_std, seed=seed)
    X_train, y_train = generator.generate_dataset(N_train)
    X_test, y_test = generator.generate_dataset(N_test)
    
    print(f"\nData generation:")
    print(f"  Dimension: {d}")
    print(f"  Training samples: {N_train}")
    print(f"  Test samples: {N_test}")
    print(f"  True sparsity: {jnp.count_nonzero(generator.get_true_theta())}/{d} non-zero")
    print(f"\nTrue theta_0: {generator.get_true_theta()}")
    
    print(f"\n" + "=" * 70)
    print(f"Method 1: Coordinate Descent")
    print("=" * 70)
    
    solver_cd = LassoSolver(method='cd', lambda_=lambda_, tol=tol, max_iter=max_iter)
    results_cd = evaluate_solver(solver_cd, X_train, y_train, X_test, y_test, generator.get_true_theta())
    
    db.insert_result('cd', d, N_train, sparsity, max_iter, tol, lambda_, results_cd)
    
    print(f"\nEstimated theta: {results_cd['coef']}")
    print(f"Estimated sparsity: {results_cd['sparsity']}/{d} non-zero")
    print(f"Time: {results_cd['time']:.4f}s (n_iter: {results_cd['n_iter']})")
    print(f"\nPerformance:")
    print(f"  Training MSE: {results_cd['train_mse']:.6f}")
    print(f"  Test MSE: {results_cd['test_mse']:.6f}")
    print(f"  Parameter estimation error (L2): {results_cd['param_error']:.6f}")
    
    print(f"\n" + "=" * 70)
    print(f"Method 2: Gradient Descent on Full Lasso Loss (exact=False)")
    print("=" * 70)
    
    solver_gd = LassoSolver(method='gd', lambda_=lambda_, tol=tol, max_iter=max_iter)
    results_gd = evaluate_solver(solver_gd, X_train, y_train, X_test, y_test, generator.get_true_theta(), 
                                  fit_kwargs={'lr': lr, 'exact': False})
    
    db.insert_result('gd', d, N_train, sparsity, max_iter, tol, lambda_, results_gd, lr=lr)
    
    print(f"\nEstimated theta: {results_gd['coef']}")
    print(f"Estimated sparsity: {results_gd['sparsity']}/{d} non-zero")
    print(f"Time: {results_gd['time']:.4f}s (n_iter: {results_gd['n_iter']})")
    print(f"\nPerformance:")
    print(f"  Training MSE: {results_gd['train_mse']:.6f}")
    print(f"  Test MSE: {results_gd['test_mse']:.6f}")
    print(f"  Parameter estimation error (L2): {results_gd['param_error']:.6f}")
    
    print(f"\n" + "=" * 70)
    print(f"Method 3: Proximal Gradient Descent / ISTA (exact=True)")
    print("=" * 70)
    
    solver_gd_exact = LassoSolver(method='gd', lambda_=lambda_, tol=tol, max_iter=max_iter)
    results_gd_exact = evaluate_solver(solver_gd_exact, X_train, y_train, X_test, y_test, generator.get_true_theta(),
                                       fit_kwargs={'lr': lr, 'exact': True})
    
    db.insert_result('gd_exact', d, N_train, sparsity, max_iter, tol, lambda_, results_gd_exact, lr=lr)
    
    print(f"\nEstimated theta: {results_gd_exact['coef']}")
    print(f"Estimated sparsity: {results_gd_exact['sparsity']}/{d} non-zero")
    print(f"Time: {results_gd_exact['time']:.4f}s (n_iter: {results_gd_exact['n_iter']})")
    print(f"\nPerformance:")
    print(f"  Training MSE: {results_gd_exact['train_mse']:.6f}")
    print(f"  Test MSE: {results_gd_exact['test_mse']:.6f}")
    print(f"  Parameter estimation error (L2): {results_gd_exact['param_error']:.6f}")
    
    print(f"\n" + "=" * 70)
    print(f"Method 4: Fully JIT-compiled Proximal GD (exact=True)")
    print("=" * 70)
    
    solver_jit = LassoSolver(method='jit_gd', lambda_=lambda_, tol=tol, max_iter=max_iter)
    results_jit = evaluate_solver(solver_jit, X_train, y_train, X_test, y_test, generator.get_true_theta(),
                                   fit_kwargs={'lr': lr, 'exact': True})
    
    db.insert_result('jit_gd_exact', d, N_train, sparsity, max_iter, tol, lambda_, results_jit, lr=lr)
    
    print(f"\nEstimated theta: {results_jit['coef']}")
    print(f"Estimated sparsity: {results_jit['sparsity']}/{d} non-zero")
    print(f"Time: {results_jit['time']:.4f}s (n_iter: {results_jit['n_iter']})")
    print(f"\nPerformance:")
    print(f"  Training MSE: {results_jit['train_mse']:.6f}")
    print(f"  Test MSE: {results_jit['test_mse']:.6f}")
    print(f"  Parameter estimation error (L2): {results_jit['param_error']:.6f}")
    
    print(f"\n" + "=" * 70)
    print(f"Method 5: Fully JIT-compiled GD on Full Loss (exact=False)")
    print("=" * 70)
    
    solver_jit_full = LassoSolver(method='jit_gd', lambda_=lambda_, tol=tol, max_iter=max_iter)
    results_jit_full = evaluate_solver(solver_jit_full, X_train, y_train, X_test, y_test, generator.get_true_theta(),
                                       fit_kwargs={'lr': lr, 'exact': False})
    
    db.insert_result('jit_gd', d, N_train, sparsity, max_iter, tol, lambda_, results_jit_full, lr=lr)
    
    print(f"\nEstimated theta: {results_jit_full['coef']}")
    print(f"Estimated sparsity: {results_jit_full['sparsity']}/{d} non-zero")
    print(f"Time: {results_jit_full['time']:.4f}s (n_iter: {results_jit_full['n_iter']})")
    print(f"\nPerformance:")
    print(f"  Training MSE: {results_jit_full['train_mse']:.6f}")
    print(f"  Test MSE: {results_jit_full['test_mse']:.6f}")
    print(f"  Parameter estimation error (L2): {results_jit_full['param_error']:.6f}")
    
    print(f"\n" + "=" * 70)
    print(f"Baseline: sklearn.linear_model.Lasso")
    print("=" * 70)
    
    sklearn_lasso = Lasso(alpha=lambda_, fit_intercept=False, max_iter=max_iter, tol=tol)
    results_sklearn = evaluate_solver(sklearn_lasso, X_train, y_train, X_test, y_test, generator.get_true_theta())
    
    db.insert_result('sklearn', d, N_train, sparsity, max_iter, tol, lambda_, results_sklearn)
    
    print(f"\nEstimated theta: {results_sklearn['coef']}")
    print(f"Estimated sparsity: {results_sklearn['sparsity']}/{d} non-zero")
    print(f"Time: {results_sklearn['time']:.4f}s (n_iter: {results_sklearn['n_iter']})")
    print(f"\nPerformance:")
    print(f"  Training MSE: {results_sklearn['train_mse']:.6f}")
    print(f"  Test MSE: {results_sklearn['test_mse']:.6f}")
    print(f"  Parameter estimation error (L2): {results_sklearn['param_error']:.6f}")
    
    print(f"\n" + "=" * 70)
    print(f"Baseline: sklearn.linear_model.LassoLars")
    print("=" * 70)
    
    sklearn_lars = LassoLars(alpha=lambda_, fit_intercept=False, max_iter=max_iter)
    results_lars = evaluate_solver(sklearn_lars, X_train, y_train, X_test, y_test, generator.get_true_theta())
    
    db.insert_result('sklearn_lasso_lars', d, N_train, sparsity, max_iter, tol, lambda_, results_lars)
    
    print(f"\nEstimated theta: {results_lars['coef']}")
    print(f"Estimated sparsity: {results_lars['sparsity']}/{d} non-zero")
    print(f"Time: {results_lars['time']:.4f}s (n_iter: {results_lars['n_iter']})")
    print(f"\nPerformance:")
    print(f"  Training MSE: {results_lars['train_mse']:.6f}")
    print(f"  Test MSE: {results_lars['test_mse']:.6f}")
    print(f"  Parameter estimation error (L2): {results_lars['param_error']:.6f}")
    
    db.close()
    print(f"\n✓ Results saved to database: lasso_experiments.db")
    
    print(f"\n" + "=" * 70)
    print(f"Comparison of All Methods")
    print("=" * 70)
    print(f"\nTest MSE:")
    print(f"  Coordinate Descent: {results_cd['test_mse']:.6f}")
    print(f"  GD (full loss, exact=False): {results_gd['test_mse']:.6f}")
    print(f"  Proximal GD (exact=True): {results_gd_exact['test_mse']:.6f}")
    print(f"  Fully JIT Proximal GD (exact=True): {results_jit['test_mse']:.6f}")
    print(f"  Fully JIT GD (full loss, exact=False): {results_jit_full['test_mse']:.6f}")
    print(f"  sklearn Lasso (baseline): {results_sklearn['test_mse']:.6f}")
    print(f"  sklearn LassoLars (baseline): {results_lars['test_mse']:.6f}")
    print(f"\nParameter Estimation Error (L2):")
    print(f"  Coordinate Descent: {results_cd['param_error']:.6f}")
    print(f"  GD (full loss, exact=False): {results_gd['param_error']:.6f}")
    print(f"  Proximal GD (exact=True): {results_gd_exact['param_error']:.6f}")
    print(f"  Fully JIT Proximal GD (exact=True): {results_jit['param_error']:.6f}")
    print(f"  Fully JIT GD (full loss, exact=False): {results_jit_full['param_error']:.6f}")
    print(f"  sklearn Lasso (baseline): {results_sklearn['param_error']:.6f}")
    print(f"  sklearn LassoLars (baseline): {results_lars['param_error']:.6f}")
    print(f"\nSparsity (non-zero coefficients):")
    print(f"  True theta: {jnp.count_nonzero(generator.get_true_theta())}/{d}")
    print(f"  Coordinate Descent: {results_cd['sparsity']}/{d}")
    print(f"  GD (full loss, exact=False): {results_gd['sparsity']}/{d}")
    print(f"  Proximal GD (exact=True): {results_gd_exact['sparsity']}/{d}")
    print(f"  Fully JIT Proximal GD (exact=True): {results_jit['sparsity']}/{d}")
    print(f"  Fully JIT GD (full loss, exact=False): {results_jit_full['sparsity']}/{d}")
    print(f"  sklearn Lasso (baseline): {results_sklearn['sparsity']}/{d}")
    print(f"  sklearn LassoLars (baseline): {results_lars['sparsity']}/{d}")
    print(f"\nExecution Time:")
    print(f"  Coordinate Descent: {results_cd['time']:.4f}s")
    print(f"  GD (full loss, exact=False): {results_gd['time']:.4f}s")
    print(f"  Proximal GD (exact=True): {results_gd_exact['time']:.4f}s")
    print(f"  Fully JIT Proximal GD (exact=True): {results_jit['time']:.4f}s")
    print(f"  Fully JIT GD (full loss, exact=False): {results_jit_full['time']:.4f}s")
    print(f"  sklearn Lasso (baseline): {results_sklearn['time']:.4f}s")
    print(f"  sklearn LassoLars (baseline): {results_lars['time']:.4f}s")
    print(f"\nTime per iteration:")
    print(f"  Coordinate Descent: {results_cd['time_per_iter']*1000:.4f}ms/iter")
    print(f"  GD (full loss, exact=False): {results_gd['time_per_iter']*1000:.4f}ms/iter")
    print(f"  Proximal GD (exact=True): {results_gd_exact['time_per_iter']*1000:.4f}ms/iter")
    print(f"  Fully JIT Proximal GD (exact=True): {results_jit['time_per_iter']*1000:.4f}ms/iter")
    print(f"  Fully JIT GD (full loss, exact=False): {results_jit_full['time_per_iter']*1000:.4f}ms/iter")
    print(f"  sklearn Lasso (baseline): {results_sklearn['time_per_iter']*1000:.4f}ms/iter")
    print(f"  sklearn LassoLars (baseline): {results_lars['time_per_iter']*1000:.4f}ms/iter")


if __name__ == "__main__":
    main()

