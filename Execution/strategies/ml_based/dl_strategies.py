"""
Deep Learning Trading Strategies
Includes LSTM, Transformer, CNN, and attention-based models
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from strategy.ml_base import BaseMLStrategy, PredictionType, MLSignal, FeatureEngineer, ModelState

# Import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as flax_nn
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


# ========== PyTorch Models ==========

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 3,
                 bidirectional: bool = True):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Attention layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last time step
        last_out = attn_out[:, -1, :]

        # FC layers
        out = self.relu(self.fc1(last_out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class TransformerModel(nn.Module):
    """Transformer model for sequence prediction"""

    def __init__(self,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 output_size: int = 3):
        super().__init__()

        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Embed input
        x = self.input_embedding(x)
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer_encoder(x)

        # Use last time step
        x = x[:, -1, :]

        # Output
        x = self.dropout(x)
        x = self.fc(x)

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class CNNModel(nn.Module):
    """1D CNN model for pattern recognition in time series"""

    def __init__(self,
                 input_size: int,
                 num_filters: List[int] = [64, 128, 256],
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout: float = 0.2,
                 output_size: int = 3):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = input_size

        # Convolutional layers
        for num_filter, kernel_size in zip(num_filters, kernel_sizes):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, num_filter, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(num_filter),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ))
            in_channels = num_filter

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters[-1], 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Transpose for Conv1d (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Global pooling
        x = self.global_pool(x).squeeze(-1)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ========== Strategy Classes ==========

class LSTMStrategy(BaseMLStrategy):
    """LSTM-based trading strategy"""

    def __init__(self,
                 prediction_type: PredictionType = PredictionType.CLASSIFICATION,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 **kwargs):
        super().__init__(
            name="LSTM",
            prediction_type=prediction_type,
            **kwargs
        )

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMStrategy")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)

    def build_model(self, input_shape: Tuple[int, ...], **kwargs):
        """Build LSTM model"""
        seq_len, input_size = input_shape

        output_size = 3 if self.prediction_type == PredictionType.CLASSIFICATION else 1

        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=output_size,
            bidirectional=self.bidirectional
        ).to(self.device)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """Train LSTM model"""
        # Reshape data for sequence input
        if X_train.ndim == 2:
            X_train = X_train.reshape(X_train.shape[0], -1, X_train.shape[-1] // self.lookback_window)

        if self.model is None:
            input_shape = X_train.shape[1:]
            self.build_model(input_shape)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train) if self.prediction_type == PredictionType.CLASSIFICATION
            else torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None:
            if X_val.ndim == 2:
                X_val = X_val.reshape(X_val.shape[0], -1, X_val.shape[-1] // self.lookback_window)
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val) if self.prediction_type == PredictionType.CLASSIFICATION
                else torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Loss and optimizer
        if self.prediction_type == PredictionType.CLASSIFICATION:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}" +
                      (f", Val Loss: {val_loss:.4f}" if val_loader else ""))

        self.training_history.append(history)
        self.state = ModelState.TRAINED

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")

        # Reshape data
        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, X.shape[-1] // self.lookback_window)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)

            if self.prediction_type == PredictionType.CLASSIFICATION:
                probs = torch.softmax(outputs, dim=1)
                return probs.cpu().numpy()
            else:
                return outputs.cpu().numpy()

    def _save_model_specific(self, path: Path):
        """Save LSTM model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional
        }, path.with_suffix('.pth'))

    def _load_model_specific(self, path: Path):
        """Load LSTM model"""
        checkpoint = torch.load(path.with_suffix('.pth'))
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.bidirectional = checkpoint['bidirectional']

        # Rebuild model (need input shape)
        # This will be set when first prediction is made
        # For now, just store the state dict
        self._checkpoint = checkpoint


class TransformerStrategy(BaseMLStrategy):
    """Transformer-based trading strategy"""

    def __init__(self,
                 prediction_type: PredictionType = PredictionType.CLASSIFICATION,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 **kwargs):
        super().__init__(
            name="Transformer",
            prediction_type=prediction_type,
            **kwargs
        )

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TransformerStrategy")

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)

    def build_model(self, input_shape: Tuple[int, ...], **kwargs):
        """Build Transformer model"""
        seq_len, input_size = input_shape
        output_size = 3 if self.prediction_type == PredictionType.CLASSIFICATION else 1

        self.model = TransformerModel(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            output_size=output_size
        ).to(self.device)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """Train Transformer model"""
        # Use same training logic as LSTM
        strategy = LSTMStrategy(
            prediction_type=self.prediction_type,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            device=str(self.device)
        )
        strategy.model = self.model
        strategy.lookback_window = self.lookback_window

        return strategy.train(X_train, y_train, X_val, y_val, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")

        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, X.shape[-1] // self.lookback_window)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)

            if self.prediction_type == PredictionType.CLASSIFICATION:
                probs = torch.softmax(outputs, dim=1)
                return probs.cpu().numpy()
            else:
                return outputs.cpu().numpy()

    def _save_model_specific(self, path: Path):
        """Save Transformer model"""
        torch.save(self.model.state_dict(), path.with_suffix('.pth'))

    def _load_model_specific(self, path: Path):
        """Load Transformer model"""
        # Need to rebuild model first with correct architecture
        state_dict = torch.load(path.with_suffix('.pth'))
        if self.model is not None:
            self.model.load_state_dict(state_dict)


class CNNStrategy(BaseMLStrategy):
    """CNN-based trading strategy for pattern recognition"""

    def __init__(self,
                 prediction_type: PredictionType = PredictionType.CLASSIFICATION,
                 num_filters: List[int] = [64, 128, 256],
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 **kwargs):
        super().__init__(
            name="CNN",
            prediction_type=prediction_type,
            **kwargs
        )

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CNNStrategy")

        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)

    def build_model(self, input_shape: Tuple[int, ...], **kwargs):
        """Build CNN model"""
        seq_len, input_size = input_shape
        output_size = 3 if self.prediction_type == PredictionType.CLASSIFICATION else 1

        self.model = CNNModel(
            input_size=input_size,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout,
            output_size=output_size
        ).to(self.device)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """Train CNN model"""
        # Use same training logic as LSTM
        strategy = LSTMStrategy(
            prediction_type=self.prediction_type,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            device=str(self.device)
        )
        strategy.model = self.model
        strategy.lookback_window = self.lookback_window

        return strategy.train(X_train, y_train, X_val, y_val, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")

        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, X.shape[-1] // self.lookback_window)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)

            if self.prediction_type == PredictionType.CLASSIFICATION:
                probs = torch.softmax(outputs, dim=1)
                return probs.cpu().numpy()
            else:
                return outputs.cpu().numpy()

    def _save_model_specific(self, path: Path):
        """Save CNN model"""
        torch.save(self.model.state_dict(), path.with_suffix('.pth'))

    def _load_model_specific(self, path: Path):
        """Load CNN model"""
        state_dict = torch.load(path.with_suffix('.pth'))
        if self.model is not None:
            self.model.load_state_dict(state_dict)
