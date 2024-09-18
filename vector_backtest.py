import numpy as np
import polars as pl
import pandas as pd


class VectorBackTester:
    def __init__(self, factors_df: pl.DataFrame | pd.DataFrame):
        self.factors = factors_df
        self.TRANSACTION_COST: float = 0.06 / 100
        self.factor: np.ndarray = self.factors["factor"].to_numpy()

    def _compute_rolling_mean(self, array: np.ndarray, window_size: int) -> np.ndarray:
        rolling_array = np.array([np.nan] * len(array))
        window = np.ones(window_size) / window_size
        rolling_mean = np.convolve(array, window, mode="valid")
        rolling_array[window_size - 1 :] = rolling_mean
        return rolling_array

    def _compute_rolling_std(self, array: np.ndarray, window_size: int) -> np.ndarray:
        return np.array(
            [np.nan] * (window_size - 1)
            + [
                np.std(array[i : i + window_size], ddof=1)
                for i in range(len(array) - window_size + 1)
            ]
        )

    def _z_score_strategy_position(
        self, rolling_window: int, threshold: float
    ) -> np.ndarray:
        rolling_mean = self._compute_rolling_mean(self.factor, rolling_window)
        rolling_std = self._compute_rolling_std(self.factor, rolling_window)
        z_score = (self.factor - rolling_mean) / rolling_std

        long_entry = (z_score > threshold).astype(int)
        long_exit = (z_score <= 0).astype(int)

        short_entry = (z_score < -1 * threshold).astype(int) * -1
        short_exit = (z_score >= 0).astype(int)

        position: np.ndarray = np.zeros(len(self.factor))
        for i in range(1, len(position)):
            if long_entry[i]:
                position[i] = 1
            elif short_entry[i]:
                position[i] = -1
            else:
                position[i] = position[i - 1]

            if (position[i] == 1 and long_exit[i]) or (
                position[i] == -1 and short_exit[i]
            ):
                position[i] = 0
        return position


if __name__ == "__main__":
    factors_df = pl.read_csv("factors.csv")
    backtester = VectorBackTester(factors_df)
    backtester._z_score_strategy_position(220, 2.8)
