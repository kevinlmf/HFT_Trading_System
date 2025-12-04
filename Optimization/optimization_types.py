"""
Optimization Types - Shared definitions to avoid circular imports
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Any


class ModelObjective(Enum):
    """模型目标类型"""
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_VARIANCE = "minimize_variance"
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_CVAR = "minimize_cvar"
    MAXIMIZE_UTILITY = "maximize_utility"
    MINIMIZE_TRACKING_ERROR = "minimize_tracking_error"

class AlgorithmType(Enum):
    """算法类型"""
    GRADIENT_DESCENT = "gradient_descent"
    NEWTON_METHOD = "newton_method"
    QUASI_NEWTON = "quasi_newton"
    STOCHASTIC_GRADIENT = "stochastic_gradient"
    ADAM = "adam"
    LBFGS = "lbfgs"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    MCMC = "mcmc"
    VARIATIONAL_INFERENCE = "variational_inference"

@dataclass
class AlgorithmConfig:
    """算法配置"""
    algorithm_type: AlgorithmType
    max_iterations: int = 1000
    tolerance: float = 1e-6
    learning_rate: float = 0.01
    batch_size: Optional[int] = None
    parallel: bool = True
    gpu: bool = False
    parameters: Dict[str, Any] = None

