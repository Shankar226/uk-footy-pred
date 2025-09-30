import pandas as pd
from typing import Tuple

def time_safe_split(meta: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                    val_season: int=None, test_season: int=None) -> Tuple:
    seasons = sorted(meta["season"].unique())
    if len(seasons) < 3:
        # fallback: 70/15/15 by date if too few seasons
        dates = meta["Date"].sort_values()
        t1 = dates.quantile(0.70); t2 = dates.quantile(0.85)
        tr = meta["Date"] <= t1
        va = (meta["Date"] > t1) & (meta["Date"] <= t2)
        te = meta["Date"] > t2
        return X[tr], y[tr], X[va], y[va], X[te], y[te], meta[va], meta[te]

    if test_season is None:
        test_season = seasons[-1]
    if val_season is None:
        val_season = seasons[-2]

    train_idx = meta["season"].isin([s for s in seasons if s < val_season])
    val_idx   = meta["season"]==val_season
    test_idx  = meta["season"]==test_season

    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx],
            meta[val_idx], meta[test_idx])
 
