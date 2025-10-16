import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd


class ExperimentDB:
    def __init__(self, db_path: str = "lasso_experiments.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                solver_type TEXT NOT NULL,
                num_features INTEGER NOT NULL,
                num_samples INTEGER NOT NULL,
                sparsity_of_theta REAL NOT NULL,
                max_iter INTEGER NOT NULL,
                lr REAL,
                tol REAL NOT NULL,
                lambda REAL NOT NULL,
                n_iter INTEGER,
                sparsity INTEGER,
                train_mse REAL,
                test_mse REAL,
                param_error REAL,
                time REAL,
                time_per_iter REAL,
                timestamp TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_solver_type 
            ON experiments(solver_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_num_features 
            ON experiments(num_features)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_params 
            ON experiments(num_features, num_samples, sparsity_of_theta)
        """)
        
        self.conn.commit()
    
    def insert_result(
        self,
        solver_type: str,
        num_features: int,
        num_samples: int,
        sparsity_of_theta: float,
        max_iter: int,
        tol: float,
        lambda_: float,
        results: Dict[str, Any],
        lr: Optional[float] = None
    ) -> int:
        """Upsert experiment result (updates if exists, inserts otherwise)."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT experiment_id FROM experiments
            WHERE solver_type = ? AND num_features = ? AND num_samples = ?
            AND sparsity_of_theta = ? AND max_iter = ? AND tol = ? AND lambda = ?
            AND (lr = ? OR (lr IS NULL AND ? IS NULL))
        """, (
            solver_type, num_features, num_samples, sparsity_of_theta,
            max_iter, tol, lambda_, lr, lr
        ))
        
        existing = cursor.fetchone()
        
        if existing:
            experiment_id = existing[0]
            cursor.execute("""
                UPDATE experiments
                SET n_iter = ?, sparsity = ?, train_mse = ?, test_mse = ?,
                    param_error = ?, time = ?, time_per_iter = ?, timestamp = ?
                WHERE experiment_id = ?
            """, (
                results['n_iter'],
                results['sparsity'],
                results['train_mse'],
                results['test_mse'],
                results['param_error'],
                results['time'],
                results['time_per_iter'],
                datetime.now().isoformat(),
                experiment_id
            ))
        else:
            cursor.execute("""
                INSERT INTO experiments (
                    solver_type, num_features, num_samples, sparsity_of_theta,
                    max_iter, lr, tol, lambda, n_iter, sparsity,
                    train_mse, test_mse, param_error, time, time_per_iter, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                solver_type,
                num_features,
                num_samples,
                sparsity_of_theta,
                max_iter,
                lr,
                tol,
                lambda_,
                results['n_iter'],
                results['sparsity'],
                results['train_mse'],
                results['test_mse'],
                results['param_error'],
                results['time'],
                results['time_per_iter'],
                datetime.now().isoformat()
            ))
            experiment_id = cursor.lastrowid
        
        self.conn.commit()
        return experiment_id
    
    def query(
        self,
        solver_type: Optional[str] = None,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None,
        sparsity_of_theta: Optional[float] = None,
        max_iter: Optional[int] = None,
        lr: Optional[float] = None,
        tol: Optional[float] = None,
        lambda_: Optional[float] = None
    ) -> List[Dict]:
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        
        if solver_type is not None:
            query += " AND solver_type = ?"
            params.append(solver_type)
        if num_features is not None:
            query += " AND num_features = ?"
            params.append(num_features)
        if num_samples is not None:
            query += " AND num_samples = ?"
            params.append(num_samples)
        if sparsity_of_theta is not None:
            query += " AND sparsity_of_theta = ?"
            params.append(sparsity_of_theta)
        if max_iter is not None:
            query += " AND max_iter = ?"
            params.append(max_iter)
        if lr is not None:
            query += " AND lr = ?"
            params.append(lr)
        if tol is not None:
            query += " AND tol = ?"
            params.append(tol)
        if lambda_ is not None:
            query += " AND lambda = ?"
            params.append(lambda_)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def to_dataframe(
        self,
        solver_type: Optional[str] = None,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None
    ) -> pd.DataFrame:
        results = self.query(
            solver_type=solver_type,
            num_features=num_features,
            num_samples=num_samples
        )
        return pd.DataFrame(results)
    
    def delete_experiment(self, experiment_id: int):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
        self.conn.commit()
    
    def clear_all(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM experiments")
        self.conn.commit()
    
    def close(self):
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

