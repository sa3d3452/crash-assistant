import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

BASE_CASHOUT = 1.7
LOW_LIMIT = 1.3

def train_and_predict(data_path="data.csv"):
    df = pd.read_csv(data_path)

    # Features
    df["prev1"] = df["value"].shift(1)
    df["prev2"] = df["value"].shift(2)
    df["prev3"] = df["value"].shift(3)

    # LOW streak
    low = df["value"] < LOW_LIMIT
    df["low_streak"] = low.groupby((low != low.shift()).cumsum()).cumcount() + 1
    df.loc[~low, "low_streak"] = 0

    # Volatility & trend
    df["volatility"] = df["value"].rolling(5).std()
    df["trend"] = df["value"].rolling(5).mean()

    # Target
    df["safe_next"] = (df["value"].shift(-1) >= BASE_CASHOUT).astype(int)

    df.dropna(inplace=True)

    X = df[["prev1", "prev2", "prev3", "low_streak", "volatility", "trend"]]
    y = df["safe_next"]

    model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    model.fit(X, y)

    last = X.iloc[-1:].values
    prediction = model.predict(last)[0]
    confidence = max(model.predict_proba(last)[0])

    return prediction, confidence, df
