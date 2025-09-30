import pandas as pd
from pathlib import Path

# Columns we try to keep if present in each season CSV.
FD_KEEP_COLS = [
    "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A", "PSH", "PSD", "PSA"
]

# Football-Data seasons use mixed date formats; handle common ones.
DATE_FORMATS = ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]

def _parse_date(s):
    """Robust date parser for Football-Data CSVs."""
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(s, format=fmt, dayfirst=True, errors="raise")
        except Exception:
            pass
    # final fallback
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def load_football_data(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and unify Football-Data.co.uk season CSVs from raw_dir (recursively).
    Keeps English divisions E0..E3 and standardises a 'result' column (H/D/A).
    """
    raw_path = Path(raw_dir)
    files = list(raw_path.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSVs found under {raw_path}. "
            "Download season files (e.g., E0.csv/E1.csv) into data/raw/."
        )

    frames = []
    for f in files:
        # Files are usually Latin-1; keep errors='ignore' to avoid hard stops on odd chars
        df = pd.read_csv(f, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]  # normalise headers
        keep = [c for c in FD_KEEP_COLS if c in df.columns]
        if not keep:
            # If a file is weirdly formatted, skip rather than crash
            continue
        df = df[keep].copy()
        # Parse dates
        df["Date"] = df["Date"].apply(_parse_date)
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
        frames.append(df)

    if not frames:
        raise RuntimeError("Found CSVs but none had expected columns; check the files in data/raw/.")

    data = pd.concat(frames, ignore_index=True)

    # Keep English top tiers
    if "Div" in data.columns:
        data = data[data["Div"].isin(["E0", "E1", "E2", "E3"])]

    # Standard label
    data["result"] = data["FTR"].map({"H": "H", "D": "D", "A": "A"})
    # Ensure canonical dtypes
    data["HomeTeam"] = data["HomeTeam"].astype(str)
    data["AwayTeam"] = data["AwayTeam"].astype(str)

    return data.reset_index(drop=True)
