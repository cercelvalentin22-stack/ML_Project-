"""
AI Live Market Trend Predictor

This version does NOT use the old 2018 CSV.
It downloads updated market data directly from Yahoo Finance with yfinance and
uses data from 1994 until the latest available trading day.

Run:
    pip install -r requirements.txt
    python src/live_market_ai_project.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date, timedelta
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

LIVE_TICKERS = {
    "spx": "^GSPC",      # S&P 500
    "dax": "^GDAXI",     # German DAX
    "ftse": "^FTSE",     # FTSE 100
    "nikkei": "^N225",   # Nikkei 225
}

START_DATE = "1994-01-01"


def download_live_data(start_date: str = START_DATE) -> pd.DataFrame:
    """Download updated market index data from Yahoo Finance using yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run: pip install -r requirements.txt") from exc

    # Yahoo Finance end date is exclusive, so tomorrow ensures today's/latest trading day is included if available.
    end_date = (date.today() + timedelta(days=1)).isoformat()

    frames = []
    for name, ticker in LIVE_TICKERS.items():
        downloaded = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if downloaded.empty:
            raise RuntimeError(f"No data downloaded for {ticker}. Check your internet connection or ticker symbol.")

        # yfinance can return either normal columns or MultiIndex columns,
        # depending on the installed yfinance/pandas version. This block
        # extracts the Close price robustly in both cases.
        if isinstance(downloaded.columns, pd.MultiIndex):
            close_data = downloaded["Close"]
            if isinstance(close_data, pd.DataFrame):
                close_series = close_data.iloc[:, 0]
            else:
                close_series = close_data
        else:
            close_series = downloaded["Close"]

        close_series = close_series.rename(name)
        frames.append(close_series)

    df = pd.concat(frames, axis=1).dropna()
    df.index.name = "date"
    df = df.reset_index()

    # Defensive cleanup in case pandas/yfinance returns unexpected labels.
    if "date" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Save the exact live dataset used for reproducibility.
    df.to_csv(DATA_DIR / "latest_downloaded_market_data.csv", index=False)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML features from financial time-series data."""
    data = df.copy().sort_values("date").reset_index(drop=True)
    index_cols = ["spx", "dax", "ftse", "nikkei"]

    for col in index_cols:
        data[f"{col}_return"] = data[col].pct_change()

    for col in index_cols:
        for lag in [1, 2, 3, 5, 10]:
            data[f"{col}_return_lag_{lag}"] = data[f"{col}_return"].shift(lag)

    data["spx_ma_5"] = data["spx"].rolling(5).mean()
    data["spx_ma_20"] = data["spx"].rolling(20).mean()
    data["spx_ma_50"] = data["spx"].rolling(50).mean()
    data["spx_volatility_5"] = data["spx_return"].rolling(5).std()
    data["spx_volatility_20"] = data["spx_return"].rolling(20).std()
    data["spx_momentum_5"] = data["spx"] / data["spx"].shift(5) - 1
    data["spx_momentum_20"] = data["spx"] / data["spx"].shift(20) - 1
    data["spx_ma_ratio_5_20"] = data["spx_ma_5"] / data["spx_ma_20"] - 1
    data["spx_ma_ratio_20_50"] = data["spx_ma_20"] / data["spx_ma_50"] - 1

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek

    data["spx_up_tomorrow"] = (data["spx"].shift(-1) > data["spx"]).astype(int)
    return data.dropna().reset_index(drop=True)


def make_visualisations(data: pd.DataFrame) -> None:
    """Save project visualisations with real dates on the x-axis."""
    plt.figure(figsize=(12, 5))
    plt.plot(data["date"], data["spx"])
    plt.title("S&P 500 Historical Index Level")
    plt.xlabel("Date")
    plt.ylabel("S&P 500 close")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_spx_history_with_dates.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(data["date"], data["spx_ma_5"], label="5-day moving average")
    plt.plot(data["date"], data["spx_ma_20"], label="20-day moving average")
    plt.plot(data["date"], data["spx_ma_50"], label="50-day moving average")
    plt.title("S&P 500 Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Index level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_spx_moving_averages.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(data["date"], data["spx_volatility_20"])
    plt.title("S&P 500 20-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_spx_rolling_volatility.png", dpi=150)
    plt.close()

    corr_cols = ["spx_return", "dax_return", "ftse_return", "nikkei_return"]
    corr = data[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im)
    plt.title("Correlation Between Global Market Returns")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_return_correlation_heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(data["spx_return"], bins=60)
    plt.title("Distribution of S&P 500 Daily Returns")
    plt.xlabel("Daily return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_spx_return_distribution.png", dpi=150)
    plt.close()


def train_and_evaluate(data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Train candidate models using chronological train/test split."""
    feature_cols = [col for col in data.columns if col not in ["date", "spx_up_tomorrow"]]
    X = data[feature_cols]
    y = data["spx_up_tomorrow"]

    split_index = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    test_dates = data["date"].iloc[split_index:].reset_index(drop=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), feature_cols)
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    fitted_models = {}
    predictions_by_model = {}

    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
        })
        fitted_models[name] = pipeline
        predictions_by_model[name] = pred

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    best_model_name = str(results_df.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    best_predictions = predictions_by_model[best_model_name]

    (OUTPUT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, best_predictions, zero_division=0), encoding="utf-8"
    )

    cm = confusion_matrix(y_test, best_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down/Flat", "Up"])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_confusion_matrix.png", dpi=150)
    plt.close()

    prediction_df = pd.DataFrame({
        "date": test_dates,
        "actual_spx_up_tomorrow": y_test.reset_index(drop=True),
        "predicted_spx_up_tomorrow": best_predictions,
    })
    prediction_df.to_csv(OUTPUT_DIR / "test_predictions_with_dates.csv", index=False)

    last = prediction_df.tail(120)
    plt.figure(figsize=(12, 5))
    plt.plot(last["date"], last["actual_spx_up_tomorrow"], label="Actual", marker="o", linewidth=1)
    plt.plot(last["date"], last["predicted_spx_up_tomorrow"], label="Predicted", marker="x", linewidth=1)
    plt.title("Actual vs Predicted Market Direction, Last 120 Test Days")
    plt.xlabel("Date")
    plt.ylabel("Direction: 1 = Up, 0 = Down/Flat")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_actual_vs_predicted_dates.png", dpi=150)
    plt.close()

    joblib.dump(best_model, OUTPUT_DIR / "best_live_market_direction_model.joblib")
    return results_df, best_model_name


def write_report(raw: pd.DataFrame, data: pd.DataFrame, results: pd.DataFrame, best_model_name: str) -> None:
    """Create a concise Markdown report for the university submission."""
    start = raw["date"].min().date()
    end = raw["date"].max().date()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    best = results.iloc[0]

    report = f"""# AI-Based Live Financial Market Trend Prediction Report

Generated on: **{generated}**

## 1. Project objective

The objective of this project is to build an AI model that predicts whether the **S&P 500** will close higher on the next trading day. The project is relevant for Digital Business and Innovation because financial institutions use AI to support market monitoring, risk analysis, and decision-making.

## 2. Data source and date coverage

Data source used in this run: **Yahoo Finance via yfinance**

Date range used: **{start} to {end}**

Important: this project does **not** use the old `Index2018.csv` file. Every time the script is run, it downloads updated market data and regenerates the dataset, plots, model results, predictions, and report.

## 3. Machine learning task

This is a **supervised learning binary classification** problem.

Target variable: `spx_up_tomorrow`

- `1` means the S&P 500 closes higher on the next trading day.
- `0` means the S&P 500 closes lower or remains flat on the next trading day.

## 4. Feature engineering

The model creates business and financial indicators from raw price data:

- daily returns
- lagged returns
- moving averages
- volatility
- momentum
- moving-average ratios
- calendar variables such as year, month, and day of week

## 5. Train/test strategy

The data is split chronologically. Older observations are used for training and newer observations are used for testing. This avoids using future information to predict the past.

Total usable observations after feature engineering: **{len(data):,}**

## 6. Model comparison

{results.to_markdown(index=False)}

Best model in this run: **{best_model_name}**

Best F1-score: **{best['f1_score']:.3f}**

## 7. Business interpretation

The model should not be interpreted as a guaranteed trading system. Instead, it is a decision-support prototype that demonstrates how AI can transform updated financial market data into a directional risk signal. In a real organization, this could support market dashboards, risk monitoring, or investment research.

## 8. Limitations

Financial markets are noisy and affected by news, macroeconomic indicators, interest rates, and investor behavior. This model uses only historical index prices, so its predictions are limited. Future improvements could include economic indicators, news sentiment, trading volume, and model backtesting.
"""
    (OUTPUT_DIR / "project_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    raw = download_live_data()
    data = create_features(raw)
    make_visualisations(data)
    results, best_model_name = train_and_evaluate(data)
    write_report(raw, data, results, best_model_name)

    print("\nData source: Yahoo Finance via yfinance")
    print("Date range:", raw["date"].min().date(), "to", raw["date"].max().date())
    print("\nModel comparison:")
    print(results.to_string(index=False))
    print("\nSaved updated dataset to:", DATA_DIR / "latest_downloaded_market_data.csv")
    print("Saved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
