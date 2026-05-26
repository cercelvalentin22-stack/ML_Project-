"""
Streamlit web interface for the AI Live Market Trend Predictor.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Allow importing from the local src/ folder when the app is launched from the project root.
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_market_ai_project import START_DATE, create_features, download_live_data  # noqa: E402


st.set_page_config(
    page_title="AI Live Market Trend Predictor",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_and_prepare_data(start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = download_live_data(start_date=start_date)
    feature_df = create_features(raw_df)
    return raw_df, feature_df


@st.cache_resource(show_spinner=False)
def train_models(data: pd.DataFrame):
    feature_cols = [col for col in data.columns if col not in ["date", "spx_up_tomorrow"]]
    X = data[feature_cols]
    y = data["spx_up_tomorrow"]

    split_index = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    test_dates = data["date"].iloc[split_index:].reset_index(drop=True)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                feature_cols,
            )
        ]
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            random_state=42,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    rows = []
    fitted = {}
    predictions = {}
    probabilities = {}

    for name, estimator in candidate_models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            proba = [None] * len(pred)

        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
        })
        fitted[name] = pipeline
        predictions[name] = pred
        probabilities[name] = proba

    results = pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)
    best_model_name = str(results.loc[0, "model"])
    best_model = fitted[best_model_name]
    best_predictions = predictions[best_model_name]
    best_probabilities = probabilities[best_model_name]

    prediction_df = pd.DataFrame({
        "date": test_dates,
        "actual": y_test.reset_index(drop=True),
        "predicted": best_predictions,
        "probability_up": best_probabilities,
    })

    last_features = data[feature_cols].tail(1)
    next_prediction = int(best_model.predict(last_features)[0])
    next_probability = float(best_model.predict_proba(last_features)[0, 1])

    return {
        "feature_cols": feature_cols,
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "prediction_df": prediction_df,
        "confusion_matrix": confusion_matrix(y_test, best_predictions),
        "next_prediction": next_prediction,
        "next_probability": next_probability,
    }


def direction_label(value: int) -> str:
    return "UP" if value == 1 else "DOWN / FLAT"


def main() -> None:
    st.title("📈 AI Live Market Trend Predictor")
    st.caption("A machine learning web interface that downloads live market data and predicts the next S&P 500 direction.")

    with st.sidebar:
        st.header("Settings")
        start_date = st.text_input("Start date", value=START_DATE)
        st.write("Data source: Yahoo Finance via yfinance")
        refresh = st.button("Refresh live data and retrain model")
        st.info("Run from terminal with: streamlit run streamlit_app.py")

    if refresh:
        load_and_prepare_data.clear()
        train_models.clear()

    with st.spinner("Downloading live market data and training models..."):
        raw, data = load_and_prepare_data(start_date)
        model_output = train_models(data)

    latest_date = raw["date"].max().date()
    earliest_date = raw["date"].min().date()
    latest_spx = raw["spx"].iloc[-1]
    next_prediction = model_output["next_prediction"]
    next_probability = model_output["next_probability"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest data date", str(latest_date))
    col2.metric("S&P 500 latest close", f"{latest_spx:,.2f}")
    col3.metric("Prediction", direction_label(next_prediction))
    col4.metric("Probability of UP", f"{next_probability:.1%}")

    st.subheader("1. Dataset coverage")
    st.write(
        f"The app downloaded market data from **{earliest_date}** to **{latest_date}**. "
        f"After feature engineering, the model uses **{len(data):,}** observations."
    )

    st.subheader("2. S&P 500 price history")
    price_chart = raw.set_index("date")[["spx"]]
    st.line_chart(price_chart)

    st.subheader("3. Market indicators")
    indicator_cols = ["spx_ma_5", "spx_ma_20", "spx_ma_50"]
    st.line_chart(data.set_index("date")[indicator_cols])

    st.subheader("4. Model comparison")
    st.dataframe(
        model_output["results"].style.format({
            "accuracy": "{:.3f}",
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1_score": "{:.3f}",
        }),
        use_container_width=True,
    )
    st.success(f"Best model in this run: {model_output['best_model_name']}")

    st.subheader("5. Confusion matrix")
    cm = model_output["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Down/Flat", "Actual Up"],
        columns=["Predicted Down/Flat", "Predicted Up"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("6. Recent actual vs predicted directions")
    recent_predictions = model_output["prediction_df"].tail(120).set_index("date")[["actual", "predicted"]]
    st.line_chart(recent_predictions)

    st.subheader("7. Latest prediction explanation")
    if next_prediction == 1:
        st.write(
            f"The model currently predicts that the S&P 500 is **more likely to close higher** on the next trading day. "
            f"Estimated probability of an upward move: **{next_probability:.1%}**."
        )
    else:
        st.write(
            f"The model currently predicts that the S&P 500 is **more likely to close lower or flat** on the next trading day. "
            f"Estimated probability of an upward move: **{next_probability:.1%}**."
        )

    st.warning(
        "This is an educational AI prototype, not financial advice. Financial markets are noisy and affected by news, macroeconomic indicators, and investor behavior."
    )


if __name__ == "__main__":
    main()
