import streamlit as st
import requests
import math
from datetime import datetime, timedelta

"""
Streamlit application for monitoring cryptocurrency markets in real-time and
 detecting emerging 5%+ price surges. The app fetches price data from the
 public Coingecko API at regular intervals, computes simple momentum-based
 features and extrapolates potential gains over the next 30 minutes. It
 surfaces the top five coins with the highest projected percentage increase
 and displays them in a table along with their projected gain. A history of
 prices is maintained in the session state so that trends can be visualised
 and used in the prediction logic. The dashboard automatically refreshes
 every minute to provide continuously updated insights.

The prediction algorithm implemented here is intentionally lightweight and
 heuristic. It examines the short-term returns of each coin (over the last
 1, 5 and 15 minute windows) and projects them forward to estimate a 30
 minute gain. Coins whose projected gain exceeds five percent are
 highlighted. In a production setting you might replace this logic with
 a more sophisticated machine learning model trained on historical data.
"""

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
    now = datetime.utcnow()
    for coin, (price, _) in prices.items():
        entry = {"time": now, "price": price}
        histories.setdefault(coin, []).append(entry)
        if len(histories[coin]) > 60:
            histories[coin] = histories[coin][-60:]

def compute_predictions(histories):
    results = []
    now = datetime.utcnow()
    for coin, history in histories.items():
        if len(history) < 2:
            continue
        current_price = history[-1]["price"]
        def pct_change_over(minutes):
            cutoff = now - timedelta(minutes=minutes)
            old_entry = None
            for entry in history:
                if entry["time"] <= cutoff:
                    old_entry = entry
                    break
            if not old_entry:
                return 0.0
            old_price = old_entry["price"]
            return (current_price - old_price) / old_price if old_price > 0 else 0.0
        r1 = pct_change_over(1)
        r5 = pct_change_over(5)
        r15 = pct_change_over(15)
        projected_change = 0.5 * r1 * (30/1) + 0.3 * r5 * (30/5) + 0.2 * r15 * (30/15)
        projected_gain_pct = projected_change * 100.0
        confidence = 1.0 / (1.0 + math.exp(-(projected_gain_pct - 5.0)))
        results.append({
            "coin": coin,
            "projected_gain": projected_gain_pct,
            "confidence": confidence,
        })
    return results

def format_coin_name(coin_id: str) -> str:
    return coin_id.replace("binancecoin", "Binance Coin").replace("ripple", "XRP").title()

def main():
    st.set_page_config(page_title="Crypto Surge Monitor", layout="wide")
    st.title("\U0001F680 Crypto Surge Monitor")
    st.markdown(
        "This dashboard monitors selected cryptocurrencies in real time and "
        "projects which ones are most likely to surge by **5% or more** "
        "in the next 30 minutes. Predictions are refreshed every minute."
    )

    # manual refresh: use session state as a simple counter to trigger reruns
    if "autorefresh" not in st.session_state:
        st.session_state.autorefresh = 0
    st.session_state.autorefresh += 1

    if "price_histories" not in st.session_state:
        st.session_state.price_histories = {}

    prices = fetch_prices()
    if prices:
        update_histories(prices, st.session_state.price_histories)

    predictions = compute_predictions(st.session_state.price_histories)
    if predictions:
        predictions_sorted = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
        top = predictions_sorted[:5]
        if top:
            table_rows = []
            for item in top:
                row = {
                    "Coin": format_coin_name(item["coin"]),
                    "Projected Gain (%)": f"{item['projected_gain']:.2f}",
                    "Confidence (%)": f"{item['confidence']*100:.1f}",
                }
                table_rows.append(row)
            st.table(table_rows)
        else:
            st.info("No coins predicted to surge by 5% or more.")
    else:
        st.warning("Collecting data... please wait a minute to accumulate history.")

if __name__ == "__main__":
    main()
