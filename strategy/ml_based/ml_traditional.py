"""
Traditional Machine Learning Trading Strategies
Includes Random Forest, XGBoost, LightGBM, Gradient Boosting
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from strategy.ml_base import BaseMLStrategy, PredictionType, MLSignal, FeatureEngineer

# Import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    RandomForestRegressor = None
    GradientBoostingClassifier = None
    GradientBoostingRegressor = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None


class RandomForestStrategy(BaseMLStrategy):
    """
    Random Forest based trading strategy
    Uses ensemble of decision trees for prediction
    """

    def __init__(self,
                 prediction_type: PredictionType = PredictionType.CLASSIFICATION,
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 5,
                 max_features: str = 'sqrt',
                 random_state: int = 42,
                 **kwargs):
        super().__init__(
            name="RandomForest",
            prediction_type=prediction_type,
            **kwargs
        )

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RandomForestStrategy")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def build_model(self, input_shape: Tuple[int, ...] = None, **kwargs):
        """Build Random Forest model"""
        if self.prediction_type == PredictionType.CLASSIFICATION:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=-1
            )

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """Train Random Forest model"""
        if self.model is None:
            self.build_model()

        # Train model
        self.model.fit(X_train, y_train)

        # Calculate feature importance
        feature_names = self.feature_engineer.get_feature_names()
        self.feature_importance = dict(zip(
            feature_names,
            self.model.feature_importances_
        ))

        # Evaluate on validation set
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val) if X_val is not None else None

        training_result = {
            'train_score': train_score,
            'val_score': val_score,
            'n_estimators': self.n_estimators,
            'feature_importance': self.feature_importance
        }

        self.training_history.append(training_result)
        self.state = self.state.TRAINED

        return training_result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")

        if self.prediction_type == PredictionType.CLASSIFICATION:
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)

    def _save_model_specific(self, path: Path):
        """Save Random Forest model"""
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.model, f)

    def _load_model_specific(self, path: Path):
        """Load Random Forest model"""
        with open(path.with_suffix('.pkl'), 'rb') as f:
            self.model = pickle.load(f)


class XGBoostStrategy(BaseMLStrategy):
    """
    XGBoost based trading strategy
    Uses gradient boosting with advanced regularization
    """

    def __init__(self,
                 prediction_type: PredictionType = PredictionType.CLASSIFICATION,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 gamma: float = 0.0,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(
            name="XGBoost",
            prediction_type=prediction_type,
            **kwargs
        )

        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for XGBoostStrategy")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def build_model(self, input_shape: Tuple[int, ...] = None, **kwargs):
        """Build XGBoost model"""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist'
        }

        if self.prediction_type == PredictionType.CLASSIFICATION:
            params['objective'] = 'multi:softprob'
            params['num_class'] = 3
            self.model = xgb.XGBClassifier(**params)
        else:
            params['objective'] = 'reg:squarederror'
            self.model = xgb.XGBRegressor(**params)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              early_stopping_rounds: int = 10,
              **kwargs) -> Dict[str, Any]:
        """Train XGBoost model"""
        if self.model is None:
            self.build_model()

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )

        # Calculate feature importance
        feature_names = self.feature_engineer.get_feature_names()
        importance_values = self.model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importance_values))

        # Get training results
        results = self.model.evals_result()
        training_result = {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'training_results': results,
            'feature_importance': self.feature_importance
        }

        self.training_history.append(training_result)
        self.state = self.state.TRAINED

        return training_result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")

        if self.prediction_type == PredictionType.CLASSIFICATION:
            return self.model.predict_proba(X)
        else:
            predictions = self.model.predict(X)
            return predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions

    def _save_model_specific(self, path: Path):
        """Save XGBoost model"""
        self.model.save_model(str(path.with_suffix('.xgb')))

    def _load_model_specific(self, path: Path):
        """Load XGBoost model"""
        if self.model is None:
            self.build_model()
        self.model.load_model(str(path.with_suffix('.xgb')))


class LightGBMStrategy(BaseMLStrategy):
    """
    LightGBM based trading strategy
    Fast gradient boosting optimized for large datasets
    """

    def __init__(self,
                 prediction_type: PredictionType = PredictionType.CLASSIFICATION,
                 n_estimators: int = 100,
                 max_depth: int = -1,
                 learning_rate: float = 0.1,
                 num_leaves: int = 31,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 0.0,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(
            name="LightGBM",
            prediction_type=prediction_type,
            **kwargs
        )

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is required for LightGBMStrategy")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def build_model(self, input_shape: Tuple[int, ...] = None, **kwargs):
        """Build LightGBM model"""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }

        if self.prediction_type == PredictionType.CLASSIFICATION:
            params['objective'] = 'multiclass'
            params['num_class'] = 3
            self.model = lgb.LGBMClassifier(**params)
        else:
            params['objective'] = 'regression'
            self.model = lgb.LGBMRegressor(**params)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              early_stopping_rounds: int = 10,
              **kwargs) -> Dict[str, Any]:
        """Train LightGBM model"""
        if self.model is None:
            self.build_model()

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set if X_val is not None else None,
            callbacks=[lgb.early_stopping(early_stopping_rounds)] if X_val is not None else None
        )

        # Calculate feature importance
        feature_names = self.feature_engineer.get_feature_names()
        importance_values = self.model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importance_values))

        # Get best iteration
        best_iteration = self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else self.n_estimators

        training_result = {
            'best_iteration': best_iteration,
            'feature_importance': self.feature_importance
        }

        self.training_history.append(training_result)
        self.state = self.state.TRAINED

        return training_result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")

        if self.prediction_type == PredictionType.CLASSIFICATION:
            return self.model.predict_proba(X)
        else:
            predictions = self.model.predict(X)
            return predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions

    def _save_model_specific(self, path: Path):
        """Save LightGBM model"""
        self.model.booster_.save_model(str(path.with_suffix('.lgb')))

    def _load_model_specific(self, path: Path):
        """Load LightGBM model"""
        if self.model is None:
            self.build_model()
        self.model = lgb.Booster(model_file=str(path.with_suffix('.lgb')))


class GradientBoostingStrategy(BaseMLStrategy):
    """
    Sklearn Gradient Boosting based trading strategy
    Traditional gradient boosting implementation
    """

    def __init__(self,
                 prediction_type: PredictionType = PredictionType.CLASSIFICATION,
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 learning_rate: float = 0.1,
                 subsample: float = 1.0,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(
            name="GradientBoosting",
            prediction_type=prediction_type,
            **kwargs
        )

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for GradientBoostingStrategy")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def build_model(self, input_shape: Tuple[int, ...] = None, **kwargs):
        """Build Gradient Boosting model"""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }

        if self.prediction_type == PredictionType.CLASSIFICATION:
            self.model = GradientBoostingClassifier(**params)
        else:
            self.model = GradientBoostingRegressor(**params)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """Train Gradient Boosting model"""
        if self.model is None:
            self.build_model()

        # Train model
        self.model.fit(X_train, y_train)

        # Calculate feature importance
        feature_names = self.feature_engineer.get_feature_names()
        self.feature_importance = dict(zip(
            feature_names,
            self.model.feature_importances_
        ))

        # Evaluate on validation set
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val) if X_val is not None else None

        training_result = {
            'train_score': train_score,
            'val_score': val_score,
            'feature_importance': self.feature_importance
        }

        self.training_history.append(training_result)
        self.state = self.state.TRAINED

        return training_result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")

        if self.prediction_type == PredictionType.CLASSIFICATION:
            return self.model.predict_proba(X)
        else:
            predictions = self.model.predict(X)
            return predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions

    def _save_model_specific(self, path: Path):
        """Save Gradient Boosting model"""
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.model, f)

    def _load_model_specific(self, path: Path):
        """Load Gradient Boosting model"""
        with open(path.with_suffix('.pkl'), 'rb') as f:
            self.model = pickle.load(f)


# Ensemble strategy combining multiple ML models
class EnsembleMLStrategy(BaseMLStrategy):
    """
    Ensemble strategy combining multiple ML models
    Uses weighted voting or stacking
    """

    def __init__(self,
                 strategies: List[BaseMLStrategy],
                 weights: Optional[List[float]] = None,
                 voting_method: str = 'soft',  # 'soft' or 'hard'
                 **kwargs):
        super().__init__(
            name="EnsembleML",
            prediction_type=strategies[0].prediction_type,
            **kwargs
        )

        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        self.voting_method = voting_method

        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]

    def build_model(self, input_shape: Tuple[int, ...] = None, **kwargs):
        """Build all sub-models"""
        for strategy in self.strategies:
            strategy.build_model(input_shape, **kwargs)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """Train all sub-models"""
        training_results = {}

        for i, strategy in enumerate(self.strategies):
            print(f"Training {strategy.name}...")
            result = strategy.train(X_train, y_train, X_val, y_val, **kwargs)
            training_results[strategy.name] = result

        self.state = self.state.TRAINED
        return training_results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []

        for strategy in self.strategies:
            pred = strategy.predict(X)
            predictions.append(pred)

        if self.voting_method == 'soft':
            # Weighted average of probabilities/values
            ensemble_pred = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                ensemble_pred += pred * weight
        else:
            # Hard voting (majority vote for classification)
            if self.prediction_type == PredictionType.CLASSIFICATION:
                pred_classes = [np.argmax(pred, axis=1) for pred in predictions]
                ensemble_pred = np.array([
                    np.bincount([pred[i] for pred in pred_classes],
                               weights=self.weights).argmax()
                    for i in range(len(X))
                ])
            else:
                # For regression, use weighted average
                ensemble_pred = np.zeros_like(predictions[0])
                for pred, weight in zip(predictions, self.weights):
                    ensemble_pred += pred * weight

        return ensemble_pred

    def _save_model_specific(self, path: Path):
        """Save all sub-models"""
        for i, strategy in enumerate(self.strategies):
            strategy_path = path.parent / f"{path.stem}_{strategy.name}_{i}"
            strategy.save_model(str(strategy_path))

    def _load_model_specific(self, path: Path):
        """Load all sub-models"""
        for i, strategy in enumerate(self.strategies):
            strategy_path = path.parent / f"{path.stem}_{strategy.name}_{i}"
            strategy.load_model(str(strategy_path))
