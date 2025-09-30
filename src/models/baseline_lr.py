from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_lr(Xtr, ytr):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=500, multi_class="ovr"))
    ])
    pipe.fit(Xtr, ytr)
    return pipe
 
