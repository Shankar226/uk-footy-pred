import os,
import numpy as np, 
import pandas as pd, 
import tensorflow as tf
from dotenv import load_dotenv

from src.data.load_fd import load_football_data
from src.features.engineering import build_features
from src.utils.splits import time_safe_split
from src.models.baseline_lr import train_lr
from src.models.tf_mlp import build_mlp
from src.models.tf_team_embed import build_team_embed_model
from src.eval.metrics import evaluate_probs
from src.data.live_api import get_scheduled_fixtures

tf.random.set_seed(42)
np.random.seed(42)

def main():
    load_dotenv()
    # 1) Load & season tagging
    raw = load_football_data()
    raw["season"] = (raw["Date"].dt.year + (raw["Date"].dt.month>=8)).astype(int)

    # 2) Build features
    X, y, meta, df_full = build_features(raw)

    # 3) Time-safe split
    Xtr, ytr, Xva, yva, Xte, yte, meta_va, meta_te = time_safe_split(meta, X, y)

    # 4) Baseline LR
    lr = train_lr(Xtr, ytr)
    pro_va_lr = lr.predict_proba(Xva)
    pro_te_lr = lr.predict_proba(Xte)
    print("LR Val:", evaluate_probs(yva, pro_va_lr))
    print("LR Test:", evaluate_probs(yte, pro_te_lr))

    # 5) TF MLP
    mlp = build_mlp(Xtr.shape[1])
    mlp.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=10, batch_size=512, verbose=2)
    pro_te_mlp = mlp.predict(Xte, verbose=0)
    print("MLP Test:", evaluate_probs(yte, pro_te_mlp))

    # 6) TF Team Embeddings + Tabular
    team_vocab = sorted(pd.unique(pd.concat([meta["HomeTeam"], meta["AwayTeam"]])))
    adv = build_team_embed_model(team_vocab, Xtr.shape[1])
    adv.fit([Xtr, meta.loc[Xtr.index, "HomeTeam"], meta.loc[Xtr.index, "AwayTeam"]],
            ytr, validation_data=([Xva, meta_va["HomeTeam"], meta_va["AwayTeam"]], yva),
            epochs=10, batch_size=512, verbose=2)
    pro_te_adv = adv.predict([Xte, meta_te["HomeTeam"], meta_te["AwayTeam"]], verbose=0)
    print("Advanced Test:", evaluate_probs(yte, pro_te_adv))

    # 7) Pick best by Macro-F1
    scores = {
        "lr": evaluate_probs(yte, pro_te_lr),
        "mlp": evaluate_probs(yte, pro_te_mlp),
        "adv": evaluate_probs(yte, pro_te_adv)
    }
    best_name = max(scores.items(), key=lambda kv: kv[1]["macro_f1"])[0]
    print("Best model:", best_name, scores[best_name])

    os.makedirs("models", exist_ok=True)
    if best_name == "lr":
        import joblib; joblib.dump(lr, "models/best_lr.joblib")
    elif best_name == "mlp":
        mlp.save("models/best_mlp.keras")
    else:
        adv.save("models/best_adv.keras")

    # 8) Score upcoming fixtures (requires API key)
    fixtures = get_scheduled_fixtures()
    if fixtures.empty:
        print("No fixtures scored (no API key or none scheduled).")
        return

    # Minimal pre-match features for fixtures: use recent form/rest from historical matches; odds may be absent → default priors
    def recent_form(team, date):
        sub = df_full[(df_full["Date"] < pd.to_datetime(date)) &
                      ((df_full["HomeTeam"]==team) | (df_full["AwayTeam"]==team))]\
                      .sort_values("Date").tail(5)
        if sub.empty:
            return pd.Series({"form":0.5, "gd":0.0, "rest":7})
        pts=0; gd=0
        for _,r in sub.iterrows():
            if r["HomeTeam"]==team:
                pts += {"H":3,"D":1,"A":0}[r["FTR"]]
                gd  += r["FTHG"]-r["FTAG"]
            else:
                pts += {"H":0,"D":1,"A":3}[r["FTR"]]
                gd  += r["FTAG"]-r["FTHG"]
        rest = (pd.to_datetime(date) - sub["Date"].max()).days
        return pd.Series({"form":pts/15.0, "gd":gd/5.0, "rest":max(min(rest,30),0)})

    feats=[]
    for _,row in fixtures.iterrows():
        h,a,date = row["home"], row["away"], row["utcDate"]
        hf = recent_form(h, date); af = recent_form(a, date)
        rec = {
            "pH": 1/3, "pD": 1/3, "pA": 1/3, "overround": 3.0,  # default if live odds not wired
            "home_form": hf["form"], "away_form": af["form"],
            "home_gd": hf["gd"], "away_gd": af["gd"],
            "home_rest": hf["rest"], "away_rest": af["rest"],
            "HS": 10, "HST": 3, "AS": 10, "AST": 3,
            "_home": h, "_away": a, "_date": date
        }
        feats.append(rec)
    fx = pd.DataFrame(feats)

    # Align columns with training X
    Xcols = list(X.columns)
    Xfx = fx[Xcols]

    # Predict with best model
    if best_name == "lr":
        import joblib; best = lr
        probs = best.predict_proba(Xfx)
    elif best_name == "mlp":
        probs = mlp.predict(Xfx, verbose=0)
    else:
        probs = adv.predict([Xfx, fx["_home"], fx["_away"]], verbose=0)

    idx2lab = {0:"H",1:"D",2:"A"}
    preds = probs.argmax(axis=1)
    fx["pred"] = [idx2lab[i] for i in preds]
    fx["pH_pred"]=probs[:,0]; fx["pD_pred"]=probs[:,1]; fx["pA_pred"]=probs[:,2]

    os.makedirs("outputs", exist_ok=True)
    fx[["_date","_home","_away","pred","pH_pred","pD_pred","pA_pred"]].to_csv("outputs/fixtures_predictions.csv", index=False)
    print("Saved predictions -> outputs/fixtures_predictions.csv")

if __name__ == "__main__":
    main()
 
