# strategy/pairs_trading.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

@dataclass
class PairState:
    symbols: Tuple[str, str]
    spread_mean: float
    spread_std: float
    zscore: float
    correlation: float

class PairsTradingStrategy:
    def __init__(
        self,
        lookback_period: int = 60,
        min_corr: float = 0.8,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
    ):
        self.lookback_period = int(lookback_period)
        self.min_corr = float(min_corr)
        self.z_entry = float(z_entry)
        self.z_exit = float(z_exit)

    # --------- public APIs used by strategy_comparison.py ----------
    def find_pairs(self, price_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        """return满足最低相关性的Available配对列表"""
        syms = list(price_data.keys())
        pairs = []
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                s1, s2 = syms[i], syms[j]
                df1, df2 = price_data[s1], price_data[s2]
                p1, p2 = self._align_price_series(df1, df2)
                if p1 is None or p2 is None:
                    continue
                corr = p1.tail(self.lookback_period).corr(p2.tail(self.lookback_period))
                if np.isfinite(corr) and corr >= self.min_corr:
                    pairs.append((s1, s2))
        return pairs

    def update_pair_statistics(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, PairState]:
        """calculate所有配对的 zscore / 均值 / 标准差 / 相关性，return字典"""
        states: Dict[str, PairState] = {}
        pairs = self.find_pairs(price_data)
        for s1, s2 in pairs:
            p1, p2 = self._align_price_series(price_data[s1], price_data[s2])
            if p1 is None or p2 is None:
                continue
            # 取最近 lookback_period
            p1w = p1.tail(self.lookback_period)
            p2w = p2.tail(self.lookback_period)
            if len(p1w) < 10 or len(p2w) < 10:
                continue
            # Simple价差（或可改为对冲Ratio）
            spread = p1w - p2w
            mu, sd = float(spread.mean()), float(spread.std(ddof=1))
            z = float((spread.iloc[-1] - mu) / sd) if sd > 0 else 0.0
            corr = float(p1w.corr(p2w))
            states[f"{s1}|{s2}"] = PairState((s1, s2), mu, sd, z, corr)
        return states

    def generate_signals(self, pair_states: List[PairState] | Dict[str, PairState], current_prices: Dict[str, float]):
        """基于 zscore Generate信号：|z| >= z_entry 进场，|z| <= z_exit 出场"""
        if isinstance(pair_states, dict):
            states = list(pair_states.values())
        else:
            states = pair_states
        signals = []
        for st in states:
            if st.spread_std <= 0:
                continue
            z = st.zscore
            s1, s2 = st.symbols
            # z > 0: s1 相对 s2 偏高 -> Short s1 / Long s2
            # z < 0: s1 相对 s2 偏低 -> Long s1 / Short s2
            if abs(z) >= self.z_entry:
                signals.append({"pair": (s1, s2), "z": z, "action": "enter"})
            elif abs(z) <= self.z_exit:
                signals.append({"pair": (s1, s2), "z": z, "action": "exit"})
        return signals

    def execute_pair_trade(self, signal, portfolio_value: float) -> Dict[str, float]:
        """把配对信号转成头寸Weight（Simple等权，单对不超过 10% 名义敞口）"""
        s1, s2 = signal["pair"]
        z = float(signal["z"])
        max_expo = 0.1  # 每对最多 10%
        if signal["action"] == "enter":
            # 按 z 方向对冲：Weight总额不超过 max_expo
            w = max_expo / 2.0
            if z > 0:
                # Short s1，Long s2
                return {s1: -w, s2: +w}
            else:
                # Long s1，Short s2
                return {s1: +w, s2: -w}
        else:
            return {}  # exit -> Close

    # ----------------- internal helpers -----------------
    def _align_price_series(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Align两个 DataFrame，return close 的Align Series（长度不足return None, None）"""
        def get_close(df: pd.DataFrame) -> pd.Series:
            if "close" in df.columns:
                return pd.Series(df["close"], dtype=float)
            for c in ("Close", "CLOSE"):
                if c in df.columns:
                    return pd.Series(df[c], dtype=float)
            raise ValueError("price dataframe missing 'close' column")

        s1 = get_close(df1).dropna()
        s2 = get_close(df2).dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < max(10, self.lookback_period // 2):  # ←← 修正这里的笔误
            return None, None
        return s1.loc[idx], s2.loc[idx]
