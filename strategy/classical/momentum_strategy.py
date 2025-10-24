"""
Momentum Trading Strategy
Multiple momentum signals with risk management (robust version)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


# ----------------------------
# Enums & dataclasses
# ----------------------------
class SignalType(Enum):
    PRICE_MOMENTUM = "price_momentum"
    EARNINGS_MOMENTUM = "earnings_momentum"
    TECHNICAL_MOMENTUM = "technical_momentum"
    CROSS_SECTIONAL = "cross_sectional"


@dataclass
class MomentumSignal:
    signal_type: SignalType
    strength: float   # -1 .. 1
    confidence: float # 0 .. 1
    timeframe: str
    lookback_period: int


# ----------------------------
# Helpers
# ----------------------------
def _last_float(x, default: float = 0.0) -> float:
    try:
        s = pd.Series(x)
        if len(s) == 0 or not np.isfinite(s.iloc[-1]):
            return float(default)
        return float(s.iloc[-1])
    except Exception:
        return float(default)


def _ensure_col(df: pd.DataFrame, name: str, values: pd.Series) -> pd.Series:
    """Attach a column if missing and return it as Series (no copy if exists)."""
    if name not in df.columns:
        df[name] = values
    return pd.Series(df[name])


def _safe_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    out = pd.Series(s).astype(float).pct_change(periods)
    return out.replace([np.inf, -np.inf], np.nan)


# ----------------------------
# Main strategy
# ----------------------------
class MomentumStrategy:
    """
    Multi-factor momentum strategy combining:
    - Price momentum (returns over various periods)
    - Technical momentum (MACD, RSI divergence, MA cross)
    - Cross-sectional momentum (relative to universe)
    - Volatility-adjusted momentum
    """

    def __init__(
        self,
        lookback_periods: List[int] = [20, 60, 252],
        volatility_adjustment: bool = True,
        cross_sectional_ranking: bool = True,
        max_position_size: float = 0.1,
        min_momentum_threshold: float = 0.02,
    ):
        self.lookback_periods = lookback_periods
        self.volatility_adjustment = volatility_adjustment
        self.cross_sectional_ranking = cross_sectional_ranking
        self.max_position_size = float(max_position_size)
        self.min_momentum_threshold = float(min_momentum_threshold)

    # ----------------------------
    # Public API
    # ----------------------------
    def generate_signals(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, List[MomentumSignal]]:
        signals: Dict[str, List[MomentumSignal]] = {}

        # Cross-sectional (optional)
        cross_sectional_scores: Dict[str, MomentumSignal] = {}
        if self.cross_sectional_ranking and len(price_data) > 1:
            cross_sectional_scores = self._calculate_cross_sectional_momentum(price_data)

        for symbol, df in price_data.items():
            if df is None or df.empty or len(df) < max(self.lookback_periods):
                continue

            df = df.copy()
            # Ensure canonical lower-case 'close'
            if "close" not in df.columns:
                # best-effort map
                for cand in ("Close", "CLOSE"):
                    if cand in df.columns:
                        df.rename(columns={cand: "close"}, inplace=True)
                        break
            if "close" not in df.columns:
                continue  # cannot proceed without close

            symbol_signals: List[MomentumSignal] = []
            symbol_signals.extend(self._price_momentum_signals(df))
            symbol_signals.extend(self._technical_momentum_signals(df))

            if symbol in cross_sectional_scores:
                symbol_signals.append(cross_sectional_scores[symbol])

            signals[symbol] = symbol_signals

        return signals

    def calculate_position_sizes(self, signals: Dict[str, List[MomentumSignal]], portfolio_value: float) -> Dict[str, float]:
        position_weights: Dict[str, float] = {}

        for symbol, symbol_signals in signals.items():
            if not symbol_signals:
                position_weights[symbol] = 0.0
                continue

            total_strength = 0.0
            total_conf = 0.0
            for sig in symbol_signals:
                w = max(0.0, min(1.0, float(sig.confidence)))
                total_strength += float(sig.strength) * w
                total_conf += w

            if total_conf > 0:
                avg_strength = total_strength / total_conf
                # scale by max position and average confidence
                position_weight = avg_strength * self.max_position_size * (total_conf / len(symbol_signals))
            else:
                position_weight = 0.0

            position_weights[symbol] = float(np.clip(position_weight, -self.max_position_size, self.max_position_size))

        # Cap total exposure at 100%
        total_exp = sum(abs(w) for w in position_weights.values())
        if total_exp > 1.0 and total_exp > 0:
            scale = 1.0 / total_exp
            position_weights = {k: v * scale for k, v in position_weights.items()}

        return position_weights

    def get_strategy_summary(self, signals: Dict[str, List[MomentumSignal]]) -> Dict:
        all_signals = [sig for sigs in signals.values() for sig in sigs]
        summary = {
            "total_signals": len(all_signals),
            "bullish_signals": sum(1 for s in all_signals if s.strength > 0),
            "bearish_signals": sum(1 for s in all_signals if s.strength < 0),
            "avg_confidence": float(np.mean([s.confidence for s in all_signals])) if all_signals else 0.0,
            "signal_distribution": {st.value: 0 for st in SignalType},
        }
        for s in all_signals:
            summary["signal_distribution"][s.signal_type.value] += 1
        return summary

    # ----------------------------
    # Components
    # ----------------------------
    def _price_momentum_signals(self, df: pd.DataFrame) -> List[MomentumSignal]:
        signals: List[MomentumSignal] = []
        close = pd.Series(df["close"]).astype(float)

        for period in self.lookback_periods:
            if len(close) <= period:
                continue

            ret_p = _safe_pct_change(close, period).iloc[-1]
            if not np.isfinite(ret_p):
                continue

            if self.volatility_adjustment:
                vol = _safe_pct_change(close).rolling(period, min_periods=5).std().iloc[-1]
                risk_adj = ret_p / vol if (vol is not None and np.isfinite(vol) and vol > 0) else ret_p
            else:
                risk_adj = ret_p

            # squash to [-1, 1]
            strength = float(np.tanh(risk_adj * 5.0))

            # consistency: positive sum windows ratio
            win = max(1, period // 4)
            roll_ret = _safe_pct_change(close).rolling(win, min_periods=1).sum()
            pos_ratio = roll_ret.rolling(4, min_periods=1).apply(lambda x: np.mean(x > 0), raw=False).iloc[-1]
            if not np.isfinite(pos_ratio):
                pos_ratio = 0.5
            confidence = float(min(abs(strength) * pos_ratio, 1.0))

            if abs(strength) > self.min_momentum_threshold:
                signals.append(
                    MomentumSignal(
                        signal_type=SignalType.PRICE_MOMENTUM,
                        strength=float(np.clip(strength, -1, 1)),
                        confidence=float(np.clip(confidence, 0, 1)),
                        timeframe=f"{period}d",
                        lookback_period=int(period),
                    )
                )
        return signals

    def _technical_momentum_signals(self, df: pd.DataFrame) -> List[MomentumSignal]:
        signals: List[MomentumSignal] = []
        if len(df) < 50:
            return signals

        close = _ensure_col(df, "close", pd.Series(df.get("close", np.nan), dtype=float)).astype(float)

        # --- MACD ---
        macd_strength = self._calculate_macd_momentum(df, close)
        if abs(macd_strength) > 0.1:
            signals.append(
                MomentumSignal(
                    signal_type=SignalType.TECHNICAL_MOMENTUM,
                    strength=float(macd_strength),
                    confidence=0.7,
                    timeframe="12_26_9",
                    lookback_period=26,
                )
            )

        # --- RSI(14) ---
        rsi_strength = self._calculate_rsi_momentum(df, close)
        if abs(rsi_strength) > 0.1:
            signals.append(
                MomentumSignal(
                    signal_type=SignalType.TECHNICAL_MOMENTUM,
                    strength=float(rsi_strength),
                    confidence=0.6,
                    timeframe="14d_rsi",
                    lookback_period=14,
                )
            )

        # --- MA crossover (20/50) ---
        ma_strength = self._calculate_ma_crossover_momentum(df, close)
        if abs(ma_strength) > 0.1:
            signals.append(
                MomentumSignal(
                    signal_type=SignalType.TECHNICAL_MOMENTUM,
                    strength=float(ma_strength),
                    confidence=0.8,
                    timeframe="20_50_ma",
                    lookback_period=50,
                )
            )

        return signals

    # ---------- Technicals ----------
    def _calculate_macd_momentum(self, df: pd.DataFrame, close: pd.Series) -> float:
        """
        MACD strength: robust to missing columns; always use Series then take last.
        strength ∈ [-1,1]
        """
        # Ensure MACD series
        if {"macd", "macd_signal"}.issubset(df.columns):
            macd = pd.Series(df["macd"]).astype(float)
            macd_sig = pd.Series(df["macd_signal"]).astype(float)
        else:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_sig = macd.ewm(span=9, adjust=False).mean()

        if "macd_histogram" in df.columns:
            macd_hist = pd.Series(df["macd_histogram"]).astype(float)
        else:
            macd_hist = macd - macd_sig

        # Recent direction & histogram trend
        last_macd = _last_float(macd)
        last_sig = _last_float(macd_sig)
        macd_dir = 1.0 if last_macd > last_sig else -1.0

        tail = macd_hist.tail(5)
        hist_trend = float(np.sign(_last_float(macd_hist) - float(tail.mean()) if len(tail) else 0.0))

        raw_strength = macd_dir * (0.7 + 0.3 * hist_trend)

        # Normalize by recent macd range
        macd_roll = macd.rolling(50, min_periods=10)
        macd_range = float((macd_roll.max().iloc[-1] - macd_roll.min().iloc[-1])) if len(macd) else 0.0
        if macd_range > 0:
            scale = min(abs(last_macd) / (macd_range / 4.0), 1.0)
        else:
            scale = 0.5
        strength = raw_strength * scale
        return float(np.clip(strength, -1.0, 1.0))

    def _calculate_rsi_momentum(self, df: pd.DataFrame, close: pd.Series) -> float:
        """
        RSI momentum: if RSI column missing, compute RSI(14).
        """
        if "rsi" in df.columns:
            rsi = pd.Series(df["rsi"]).astype(float)
        else:
            # Compute RSI(14)
            delta = close.diff()
            gain = delta.clip(lower=0.0)
            loss = (-delta).clip(lower=0.0)
            roll = 14
            avg_gain = gain.rolling(roll, min_periods=roll).mean()
            avg_loss = loss.rolling(roll, min_periods=roll).mean()
            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi = rsi.fillna(method="bfill").fillna(method="ffill")

        rsi_ma = rsi.rolling(10, min_periods=1).mean()
        r = _last_float(rsi, 50.0)
        rma = _last_float(rsi_ma, 50.0)

        if r > 70:
            strength = (r - 70) / 30.0 if r > rma else -(r - 70) / 30.0
        elif r < 30:
            strength = -(30.0 - r) / 30.0 if r < rma else (30.0 - r) / 30.0
        else:
            strength = ((r - 50.0) / 50.0) * (1.0 if r > rma else -1.0)

        return float(np.clip(strength, -1.0, 1.0))

    def _calculate_ma_crossover_momentum(self, df: pd.DataFrame, close: pd.Series) -> float:
        """
        MA crossover momentum using SMA(20) and SMA(50).
        Missing columns are computed on the fly.
        """
        if "sma_20" in df.columns:
            sma20 = pd.Series(df["sma_20"]).astype(float)
        else:
            sma20 = close.rolling(20, min_periods=5).mean()

        if "sma_50" in df.columns:
            sma50 = pd.Series(df["sma_50"]).astype(float)
        else:
            sma50 = close.rolling(50, min_periods=10).mean()

        c = _last_float(close, default=np.nan)
        s20 = _last_float(sma20, default=np.nan)
        s50 = _last_float(sma50, default=np.nan)

        if not all(np.isfinite([c, s20, s50])) or s20 == 0 or s50 == 0:
            return 0.0

        price_vs_short = (c - s20) / s20
        price_vs_long = (c - s50) / s50
        ma_rel = (s20 - s50) / s50

        strength = (price_vs_short + price_vs_long + ma_rel) / 3.0
        return float(np.clip(strength * 2.0, -1.0, 1.0))  # amplify

    # ----------------------------
    # Cross-sectional
    # ----------------------------
    def _calculate_cross_sectional_momentum(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, MomentumSignal]:
        momentum_scores: Dict[str, float] = {}

        for symbol, df in price_data.items():
            if df is None or len(df) < 60:
                continue
            d = df.copy()
            if "close" not in d.columns:
                for cand in ("Close", "CLOSE"):
                    if cand in d.columns:
                        d.rename(columns={cand: "close"}, inplace=True)
                        break
            if "close" not in d.columns:
                continue

            close = pd.Series(d["close"]).astype(float)
            mom = _safe_pct_change(close, 60).iloc[-1]
            if not np.isfinite(mom):
                continue

            if self.volatility_adjustment:
                vol = _safe_pct_change(close).rolling(60, min_periods=10).std().iloc[-1]
                if vol is not None and np.isfinite(vol) and vol > 0:
                    mom = mom / vol
            momentum_scores[symbol] = float(mom)

        if len(momentum_scores) < 2:
            return {}

        sorted_scores = sorted(momentum_scores.items(), key=lambda x: x[1])
        n = len(sorted_scores)
        out: Dict[str, MomentumSignal] = {}

        for i, (sym, _) in enumerate(sorted_scores):
            # rank to [-1, 1]
            rank_score = (i - n / 2) / (n / 2)
            out[sym] = MomentumSignal(
                signal_type=SignalType.CROSS_SECTIONAL,
                strength=float(np.clip(rank_score, -1.0, 1.0)),
                confidence=0.8,
                timeframe="60d_cross_sectional",
                lookback_period=60,
            )
        return out
