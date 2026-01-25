# -*- coding: utf-8 -*-
"""
Seed Spectral Classification with Optional Hyperparameter Tuning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from itertools import product
import warnings
warnings.filterwarnings('ignore')


class SeedClassifier:
    """Multi-model classifier with optional hyperparameter tuning."""
    
    def __init__(self, random_state: int = 123, tune_hyperparams: bool = False, cv_folds: int = 3):
        self.random_state = random_state
        self.tune_hyperparams = tune_hyperparams
        self.cv_folds = cv_folds
        self.models = self._init_models()
        self.results = None
        self.best_params = {}
        self.scaler = None
        
    def _init_models(self) -> dict:
        """Initialize models with anti-overfitting params and tuning grids."""
        return {
            'SVM': {
                'model': SVC(kernel='rbf', C=1, probability=True),
                'params': {'C': [0.5, 1, 5], 'gamma': ['scale', 0.01]}
            },
            'LogisticRegression': {
                'model': LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=1000),
                'params': {'C': [0.05, 0.1, 0.5]}
            },
            'LightGBM': {
                'model': LGBMClassifier(
                    learning_rate=0.05, max_depth=3, n_estimators=100,
                    num_leaves=15, min_child_samples=50,
                    reg_alpha=0.5, reg_lambda=0.5,
                    subsample=0.7, colsample_bytree=0.7,
                    verbose=-1, n_jobs=1, force_row_wise=True
                ),
                'params': {
                    'max_depth': [3, 4],
                    'reg_alpha': [0.1, 0.5]
                }
            },
            'RF': {
                'model': RandomForestClassifier(
                    n_estimators=100, max_depth=5,
                    min_samples_split=20, min_samples_leaf=10,
                    max_features='sqrt',
                    n_jobs=-1, random_state=self.random_state
                ),
                'params': {
                    'max_depth': [5, 7],
                    'min_samples_leaf': [5, 10]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(
                    learning_rate=0.05, max_depth=3, n_estimators=100,
                    min_child_weight=10,
                    reg_alpha=0.5, reg_lambda=1.0,
                    subsample=0.7, colsample_bytree=0.7,
                    eval_metric='mlogloss', verbosity=0, n_jobs=1, random_state=self.random_state
                ),
                'params': {
                    'max_depth': [3, 4],
                    'reg_lambda': [0.5, 1.0]
                }
            }
        }
    
    def _prepare_data(self, data=None, X=None, y=None, feature_cols='auto', label_col='class') -> tuple:
        """Prepare and split data into train/val/test sets."""
        if X is not None and y is not None:
            X_arr, y_arr = np.array(X), np.array(y)
        elif data is not None:
            df = pd.read_csv(data) if isinstance(data, (str, Path)) else data.copy()
            if feature_cols == 'auto':
                feature_cols = [c for c in df.columns if 'reflectance_' in c.lower()]
            print(f"Using {len(feature_cols)} features")
            X_arr = df[feature_cols].values
            y_arr = df[label_col].values
        else:
            raise ValueError("Provide either (data) or (X, y)")
        
        if not np.issubdtype(y_arr.dtype, np.number):
            y_arr = LabelEncoder().fit_transform(y_arr)
        
        print(f"Dataset: {len(X_arr)} samples, {X_arr.shape[1]} features")
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_arr, y_arr, test_size=0.3, random_state=self.random_state, stratify=y_arr
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1/3, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate classification metrics."""
        return {
            'Accuracy': round(accuracy_score(y_true, y_pred), 4),
            'Kappa': round(cohen_kappa_score(y_true, y_pred), 4),
            'F1': round(f1_score(y_true, y_pred, average='macro'), 4)
        }
    
    def _grid_search_no_cv(self, model, params, X_train, y_train, X_val, y_val):
        """Grid search using validation set (no CV, fast)."""
        param_names = list(params.keys())
        param_values = list(params.values())
        
        best_score = -1
        best_params = None
        best_model = None
        
        for values in product(*param_values):
            current_params = dict(zip(param_names, values))
            m = clone(model)
            m.set_params(**current_params)
            m.fit(X_train, y_train)
            score = accuracy_score(y_val, m.predict(X_val))
            
            if score > best_score:
                best_score = score
                best_params = current_params
                best_model = m
        
        return best_model, best_params
    
    def _train_model(self, name: str, config: dict, splits: tuple) -> dict:
        """Train a single model and return metrics."""
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        start_time = time.time()
        print(f"  [{name}] Starting...", end=" ", flush=True)
        
        if self.tune_hyperparams:
            # Step 1: Tune hyperparams using validation set (no CV, fast)
            best_model, best_p = self._grid_search_no_cv(
                config['model'], config['params'], X_train, y_train, X_val, y_val
            )
            self.best_params[name] = best_p
            
            # Step 2: Retrain with best params using CV (if cv_folds > 0)
            if self.cv_folds > 0:
                # Combine train + val for CV training
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.hstack([y_train, y_val])
                
                # Clone and set best params
                best = clone(config['model'])
                best.set_params(**best_p)
                
                # CV scores for reference
                cv_scores = cross_val_score(best, X_combined, y_combined, cv=self.cv_folds, scoring='accuracy')
                print(f"CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}", end=" | ", flush=True)
                
                # Final fit on combined data
                best.fit(X_combined, y_combined)
            else:
                best = best_model
        else:
            best = config['model']
            best.fit(X_train, y_train)
            self.best_params[name] = 'default'
        
        elapsed = time.time() - start_time
        
        train_metrics = self._calc_metrics(y_train, best.predict(X_train))
        test_metrics = self._calc_metrics(y_test, best.predict(X_test))
        gap = train_metrics['Accuracy'] - test_metrics['Accuracy']
        print(f"Done! ({elapsed:.1f}s) | Train={train_metrics['Accuracy']:.4f}, Test={test_metrics['Accuracy']:.4f}, Gap={gap:.4f}")
        
        return {
            'Model': name,
            'Time(s)': round(elapsed, 1),
            **{f'Train_{k}': v for k, v in train_metrics.items()},
            **{f'Val_{k}': v for k, v in self._calc_metrics(y_val, best.predict(X_val)).items()},
            **{f'Test_{k}': v for k, v in test_metrics.items()}
        }
    
    def run(self, data=None, X=None, y=None, feature_cols='auto', label_col='class') -> pd.DataFrame:
        """Run classification with all models."""
        if self.tune_hyperparams:
            mode = f"Tuning + CV={self.cv_folds}" if self.cv_folds > 0 else "Tuning (No CV)"
        else:
            mode = "Default Params"
        
        print("="*70)
        print(f"Seed Spectral Classification ({mode})")
        print("="*70)
        
        splits = self._prepare_data(data, X, y, feature_cols, label_col)
        
        print("\nTraining models...")
        print("-"*70)
        
        total_start = time.time()
        results = []
        for i, (name, config) in enumerate(self.models.items(), 1):
            print(f"[{i}/{len(self.models)}] {name}")
            try:
                results.append(self._train_model(name, config, splits))
            except Exception as e:
                print(f"  [{name}] ERROR: {e}")
                continue
        
        total_time = time.time() - total_start
        print("-"*70)
        print(f"Total training time: {total_time:.1f}s")
        
        self.results = pd.DataFrame(results)
        print("\n" + "="*70)
        print("RESULTS SUMMARY:")
        print("="*70)
        print(self.results.to_string(index=False))
        
        if self.tune_hyperparams:
            print("\nBest Parameters:")
            for k, v in self.best_params.items():
                print(f"  {k}: {v}")
        
        return self.results
    
    def save(self, output_dir: str = 'results'):
        """Save results to CSV and Excel."""
        Path(output_dir).mkdir(exist_ok=True)
        self.results.to_csv(f'{output_dir}/model_performance.csv', index=False)
        
        with pd.ExcelWriter(f'{output_dir}/classification_results.xlsx') as writer:
            self.results.to_excel(writer, sheet_name='Performance', index=False)
            pd.DataFrame([
                {'Model': k, 'Parameters': str(v)} for k, v in self.best_params.items()
            ]).to_excel(writer, sheet_name='Best_Params', index=False)
        
        print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Seed Spectral Classification')
    parser.add_argument('data', help='Path to CSV file')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--cv', type=int, default=3, help='CV folds for final training (0 = no CV)')
    parser.add_argument('--output', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    clf = SeedClassifier(tune_hyperparams=args.tune, cv_folds=args.cv)
    clf.run(data=args.data)
    clf.save(output_dir=args.output)
