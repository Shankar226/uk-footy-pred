import pandas as pd
import numpy as np

RWIN = 5  # rolling window for form

def implied_probs(row):
    H = row.get("PSH", np.nan); D = row.get("PSD", np.nan); A = row.get("PSA", np.nan)
    if np.isnan(H) or np.isnan(D) or np.isnan(A):
        H = row.get("B365H", np.nan); D = row.get("B365D", np.nan); A = row.get("B365A", np.nan)
    if any(np.isnan([H, D, A])):
        return pd.Series({"pH":np.nan,"pD":np.nan,"pA":np.nan,"overround":np.nan})
    inv = 1/H + 1/D + 1/A
    return pd.Series({"pH":(1/H)/inv, "pD":(1/D)/inv, "pA":(1/A)/inv, "overround":inv})

def add_team_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    home = df[["Date","HomeTeam","FTHG","FTAG","FTR"]].copy()
    home["Team"]=home["HomeTeam"]; home["GF"]=home["FTHG"]; home["GA"]=home["FTAG"]
    home["pts"]=home["FTR"].map({"H":3,"D":1,"A":0})

    away = df[["Date","AwayTeam","FTAG","FTHG","FTR"]].copy()
    away["Team"]=away["AwayTeam"]; away["GF"]=away["FTAG"]; away["GA"]=away["FTHG"]
    away["pts"]=away["FTR"].map({"H":0,"D":1,"A":3})

    comb = pd.concat([home[["Date","Team","GF","GA","pts"]],
                      away[["Date","Team","GF","GA","pts"]]], ignore_index=True)
    comb = comb.sort_values(["Team","Date"])
    comb["rolling_pts"] = comb.groupby("Team")["pts"].rolling(RWIN, min_periods=1).mean().reset_index(level=0, drop=True)
    comb["rolling_gd"]  = (comb.groupby("Team")["GF"].rolling(RWIN, min_periods=1).mean().reset_index(level=0, drop=True) -
                           comb.groupby("Team")["GA"].rolling(RWIN, min_periods=1).mean().reset_index(level=0, drop=True))

    df = df.merge(comb[["Date","Team","rolling_pts","rolling_gd"]],
                  left_on=["Date","HomeTeam"], right_on=["Date","Team"], how="left")\
           .rename(columns={"rolling_pts":"home_form","rolling_gd":"home_gd"}).drop(columns=["Team"])
    df = df.merge(comb[["Date","Team","rolling_pts","rolling_gd"]],
                  left_on=["Date","AwayTeam"], right_on=["Date","Team"], how="left")\
           .rename(columns={"rolling_pts":"away_form","rolling_gd":"away_gd"}).drop(columns=["Team"])
    return df

def build_features(df: pd.DataFrame):
    probs = df.apply(implied_probs, axis=1)
    df = pd.concat([df, probs], axis=1)

    for side, col in [("HomeTeam","home_rest"), ("AwayTeam","away_rest")]:
        df = df.sort_values("Date")
        df[col] = (df.groupby(side)["Date"].diff().dt.days).fillna(7).clip(0, 30)

    df = add_team_form(df)

    feats = ["pH","pD","pA","overround","home_form","away_form","home_gd","away_gd",
             "home_rest","away_rest","HS","HST","AS","AST"]
    feats = [f for f in feats if f in df.columns]
    X = df[feats].astype(float).fillna({
        "pH":1/3,"pD":1/3,"pA":1/3,"overround":3.0,
        "HS":10,"HST":3,"AS":10,"AST":3
    }).fillna(0.0)

    y = df["result"].map({"H":0,"D":1,"A":2}).astype(int)
    meta = df[["Date","HomeTeam","AwayTeam","Div","season"]] if "season" in df.columns else df.assign(season=0)[["Date","HomeTeam","AwayTeam","Div","season"]]
    return X, y, meta, df
 
