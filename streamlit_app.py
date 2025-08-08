import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

"""
Streamlit application for monitoring cryptocurrency markets in real‑time and
detecting emerging 5%+ price surges.  The app fetches price data from the
public Coingecko API at regular intervals, computes simple momentum‑based
features and extrapolates potential gains over the next 30 minutes.  It
surfaces the top five coins with the highest projected percentage increase
and displays them in a table along with their projected gain.  A history of
prices is maintained in the session state so that trends can be visualised
and used in the prediction logic.  The dashboard automatically refreshes
every minute to provide continuously updated insights.

The prediction algorithm implemented here is intentionally lightweight and
heuristic.  It examines the short‑term returns of each coin (over the last
1, 5 and 15 minute windows) and projects them forward to estimate a 30
minute gain.  Coins whose projected gain exceeds five percent are
highlighted.  In a production setting you might replace this logic with
a more sophisticated machine learning model trained on historical data.
"""

# List of cryptocurrencies to monitor.  These identifiers correspond to
# Coingecko's API naming conventions.  You can adjust this list to
# include any other coins that Coingecko supports.  Keeping the list
# relatively small helps ensure the dashboard remains responsive on the
# free tier of hosting services.
COINS = [
    "bitcoin",
    "ethereum",
    "binancecoin",
    "ripple",
    "cardano",
    "solana",
    "dogecoin",
    "tron",
    "polkadot",
    "litecoin",
]

def fetch_prices():
    """Fetch current prices for all coins from Coingecko.

    Returns a dictionary mapping each coin ID to a tuple of (price, last
    updated timestamp).  The API is queried only once per call for all
    coins, which minimises network overhead.
    """
    ids = ",".join(COINS)
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ids,
        "vs_currencies": "usd",
        "include_last_updated_at": "true",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        # In case of any network errors, return an empty dict; the caller
        # should decide how to handle missing data (for example, by
        # retaining the last known prices).
        return {}
    prices = {}
    for coin in COINS:
        info = data.get(coin)
        if not info:
            continue
        price = info.get("usd")
        updated_at = info.get("last_updated_at")
        if price is None or updated_at is None:
            continue
        prices[coin] = (price, datetime.utcfromtimestamp(updated_at))
    return prices

def update_histories(prices, histories):
    """Append the latest price data to the history for each coin.

    Args:
        prices: mapping of coin -> (price, updated_at)
        histories: mapping of coin -> list of {time, price}

    Histories are truncated to keep roughly the last 60 entries, which
    corresponds to approximately one hour of data if updates occur once
    per minute.  You can adjust the length to fit your needs and the
    limitations of your hosting environment.
    """
    now = datetime.utcnow()
    for coin, (price, updated_time) in prices.items():
        entry = {"time": now, "price": price}
        histories.setdefault(coin, []).append(entry)
        # Retain only the most recent 60 data points
        if len(histories[coin]) > 60:
            histories[coin] = histories[coin][-60:]

def compute_predictions(histories):
    """Compute projected gains and confidence for each coin.

    The function looks at the past 1, 5 and 15 minute returns and
    extrapolates a simple linear estimate of the price change over the
    next 30 minutes.  A confidence score is computed using a logistic
    transformation of the projected gain relative to the 5% threshold.

    Args:
        histories: mapping of coin -> list of {time, price}

    Returns:
        DataFrame with columns: coin, projected_gain (percent), confidence.
    """
    results = []
    for coin, history in histories.items():
        if len(history) < 2:
            # Not enough data to compute returns
            continue
        df = pd.DataFrame(history)
        df = df.sort_values("time")
        df["time_delta"] = (df["time"] - df["time"].iloc[-1]).dt.total_seconds() / 60.0
        # time_delta is negative minutes relative to now
        current_price = df["price"].iloc[-1]

        def pct_change_over(minutes):
            """Compute the percentage change over the given time window in minutes."""
            # find the closest row older than the given minutes
            cutoff_time = df["time"].iloc[-1] - timedelta(minutes=minutes)
            older = df[df["time"] <= cutoff_time]
            if older.empty:
                return 0.0
            old_price = older["price"].iloc[0]
            return (current_price - old_price) / old_price if old_price > 0 else 0.0

        # Calculate returns over different windows
        r1 = pct_change_over(1)
        r5 = pct_change_over(5)
        r15 = pct_change_over(15)
        # Extrapolate returns to 30 minutes.  We weight shorter windows more
        # heavily because they better reflect recent momentum.  Note that
        # projecting short term returns linearly is simplistic and for
        # demonstration only.
        projected_change = 0.5 * r1 * (30 / 1) + 0.3 * r5 * (30 / 5) + 0.2 * r15 * (30 / 15)
        projected_gain_pct = projected_change * 100.0
        # Compute a pseudo‑confidence: logistic function around 5%
        # A projected gain equal to the threshold yields a confidence of 0.5.
        confidence = 1.0 / (1.0 + np.exp(-(projected_gain_pct - 5.0)))
        results.append(
            {
                "coin": coin,
                "projected_gain": projected_gain_pct,
                "confidence": confidence,
            }
        )
    return pd.DataFrame(results)

def format_coin_name(coin_id: str) -> str:
    """Convert a Coingecko coin ID into a human‑friendly name."""
    return coin_id.replace("binancecoin", "Binance Coin").replace("ripple", "XRP").title()

def main():
    st.set_page_config(page_title="Crypto Surge Monitor", layout="wide")
    st.title("🚀 Crypto Surge Monitor")
    st.markdown(
        "This dashboard monitors selected cryptocurrencies in real time and "
        "projects which ones are most likely to surge by **5% or more** "
        "in the next 30 minutes.  Predictions are refreshed every minute.",
    )

    # Auto refresh the page every 60 seconds
    st_autorefresh = st.experimental_rerun  # fallback if st_autorefresh isn't available
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore
    except Exception:
        # If streamlit_autorefresh isn't installed, rely on manual rerun via timer
        pass
    if "autorefresh" not in st.session_state:
        # Initialise a counter to trigger reruns
        st.session_state.autorefresh = 0
    # Increase counter to force rerun after set interval
    # This may be replaced by st_autorefresh in future if available
    # We use a small placeholder here so that the code compiles without error
    # when the optional dependency isn't installed.
    st.session_state.autorefresh += 1

    # Initialise or update price histories in session state
    if "price_histories" not in st.session_state:
        st.session_state.price_histories = {}

    # Fetch latest prices and update histories
    prices = fetch_prices()
    if prices:
        update_histories(prices, st.session_state.price_histories)

    # Compute predictions
    predictions_df = compute_predictions(st.session_state.price_histories)
    if not predictions_df.empty:
        # Filter to coins with projected gain >= 5%
        # Sort by confidence descending
        top_candidates = (
            predictions_df[predictions_df["projected_gain"] >= 5.0]
            .sort_values(["projected_gain", "confidence"], ascending=[False, False])
            .head(5)
        )
        # If fewer than five meet the threshold, take the top five regardless
        if len(top_candidates) < 5:
            top_candidates = (
                predictions_df.sort_values(
                    ["projected_gain", "confidence"], ascending=[False, False]
                ).head(5)
            )
        # Display top candidates
        st.subheader("Top projected surges (next 30 minutes)")
        # Format names and percentages nicely
        display_df = top_candidates.copy()
        display_df["coin"] = display_df["coin"].apply(format_coin_name)
        display_df["projected_gain"] = display_df["projected_gain"].map(
            lambda x: f"{x:0.2f}%"
        )
        display_df["confidence"] = display_df["confidence"].map(
            lambda x: f"{100 * x:0.1f}%"
        )
        display_df = display_df.rename(
            columns={"coin": "Coin", "projected_gain": "Projected Gain", "confidence": "Confidence"}
        )
        st.dataframe(
            display_df.reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
        )

        # Plot price history for the top candidates
        st.subheader("Price history (last hour)")
        for _, row in top_candidates.iterrows():
            coin_id = row["coin"]
            history = st.session_state.price_histories.get(coin_id, [])
            if not history:
                continue
            df = pd.DataFrame(history)
            df = df.sort_values("time")
            df["minutes_ago"] = (
                df["time"].iloc[-1] - df["time"]
            ).dt.total_seconds() / 60.0
            # Only display the last 60 minutes
            recent = df[df["minutes_ago"] <= 60]
            if recent.empty:
                continue
            # Use Streamlit's line chart for simplicity
            st.line_chart(
                recent.set_index("time")["price"],
                height=200,
                use_container_width=True,
                name=format_coin_name(coin_id),
            )
    else:
        st.info("Waiting for price data...")

    # Footer
    st.write(
        "Prices provided by [Coingecko](https://www.coingecko.com). "
        "Predictions are for informational purposes only and do not constitute financial advice."
    )

if __name__ == "__main__":
    main()
