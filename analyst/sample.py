"""Generates a demo dataset with realistic problems (missing values, duplicates, outliers)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_data(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    order_date = rng.choice(dates, size=n)
    region = rng.choice(["North", "South", "East", "West"], size=n, p=[0.3, 0.2, 0.3, 0.2])
    category = rng.choice(["Electronics", "Furniture", "Office Supplies", "Clothing"], size=n, p=[0.3, 0.2, 0.35, 0.15])
    base_price = {"Electronics": 320, "Furniture": 210, "Office Supplies": 25, "Clothing": 60}
    unit_price = np.array([base_price[c] for c in category]) * rng.normal(1.0, 0.15, n)
    units = rng.integers(1, 12, n)
    discount = rng.choice([0, 0.05, 0.1, 0.2], size=n, p=[0.5, 0.25, 0.15, 0.1])
    growth = 1 + (pd.to_datetime(order_date) - dates[0]).days.to_numpy() / 900  # sales trend upward
    revenue = unit_price * units * (1 - discount) * growth
    df = pd.DataFrame(
        {
            "order_id": np.arange(10001, 10001 + n),
            "order_date": pd.to_datetime(order_date),
            "region": region,
            "category": category,
            "customer_age": rng.normal(38, 11, n).round().clip(18, 80),
            "units": units,
            "unit_price": unit_price.round(2),
            "discount": discount,
            "revenue": revenue.round(2),
            "satisfaction": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.1, 0.2, 0.35, 0.3]).astype(float),
        }
    ).sort_values("order_date").reset_index(drop=True)

    # inject problems
    df.loc[rng.choice(n, 60, replace=False), "customer_age"] = np.nan
    df.loc[rng.choice(n, 110, replace=False), "satisfaction"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "region"] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "revenue"] *= 12  # outliers
    df = pd.concat([df, df.sample(18, random_state=1)], ignore_index=True)  # duplicates
    return df
