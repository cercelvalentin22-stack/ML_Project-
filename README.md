# AI Live Market Trend Predictor

This project predicts whether the S&P 500 will close higher on the next trading day using live financial market data.

Every time you run the project, it downloads updated data from Yahoo Finance using `yfinance`, from 1994 until the latest available trading day.

## Data source

The script downloads these market indexes:

- `^GSPC` = S&P 500
- `^GDAXI` = German DAX
- `^FTSE` = FTSE 100
- `^N225` = Nikkei 225

The downloaded file is saved automatically to:

```text
data/latest_downloaded_market_data.csv
```

## How to run

```bash
pip install -r requirements.txt
python src/live_market_ai_project.py
```

## Outputs generated automatically

After running the script, the `outputs/` folder will contain:

- market trend plots with dates
- moving average plots
- volatility plot
- correlation heatmap
- model comparison table
- classification report
- confusion matrix
- prediction file with dates
- final Markdown report
- trained model file

## Machine learning task

This is a supervised binary classification problem.

Target:

```text
spx_up_tomorrow
```

Meaning:

- `1` = S&P 500 closes higher tomorrow
- `0` = S&P 500 closes lower or flat tomorrow

## Models used

- Logistic Regression
- Random Forest
- Gradient Boosting

## Business relevance

The project demonstrates how AI can support financial decision-making, market monitoring, and risk analysis using continuously updated data.

## Optional web interface

This project also includes a Streamlit web interface.

Run it with:

```bash
streamlit run streamlit_app.py
```

The interface downloads live data, trains the models, shows the latest date, displays charts, compares model performance, and gives the latest next-day market direction prediction.
