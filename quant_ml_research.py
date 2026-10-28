#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install yfinance')
get_ipython().run_line_magic('pip', 'install scikit-learn')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 1. GENERATE SYNTHETIC MARKET DATA
# ============================================================

np.random.seed(42)

n = 5000
mid = np.zeros(n)
mid[0] = 100.0

returns = np.zeros(n)
order_imbalance = np.random.normal(0, 1, n)

# Toy market: small momentum effect plus random noise
for t in range(1, n):
    returns[t] = (
        0.15 * returns[t-1]
        + 0.0003 * order_imbalance[t-1]
        + np.random.normal(0, 0.01)
    )
    mid[t] = mid[t-1] * np.exp(returns[t])

df = pd.DataFrame({
    "mid": mid,
    "return": returns,
    "imbalance": order_imbalance
})

# ============================================================
# 2. CREATE REGRESSION FEATURES
# ============================================================

# Information available at time t
df["lag_return"] = df["return"].shift(1)
df["lag_imbalance"] = df["imbalance"].shift(1)
df["volatility"] = (
    df["return"].rolling(20).std().shift(1)
)

# Predict the next period's return
df["target"] = df["return"].shift(-1)

df = df.dropna().reset_index(drop=True)

features = ["lag_return", "lag_imbalance", "volatility"]

X = df[features]
y = df["target"]

# Chronological split: never randomly shuffle time-series data
split = int(0.7 * len(df))

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# ============================================================
# 3. FIT LINEAR REGRESSION
# ============================================================

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Regression results")
print("------------------")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Test MSE:", mean_squared_error(y_test, predictions))
print("Test R²:", r2_score(y_test, predictions))
print("Directional accuracy:",
      np.mean(np.sign(predictions) == np.sign(y_test)))

# ============================================================
# 4. MARKET-MAKING SIMULATION
# ============================================================

# Simulate only the unseen test period
test = df.iloc[split:].copy().reset_index(drop=True)
test["prediction"] = predictions

# Illustrative parameters; tune on training/validation data
BASE_HALF_SPREAD = 0.015
ALPHA_STRENGTH = 0.5
INVENTORY_SKEW = 0.003
MAX_INVENTORY = 10
FEE_PER_UNIT = 0.0005

cash = 0.0
inventory = 0
equity_history = []
inventory_history = []
spread_history = []
fill_count = 0

for t in range(len(test) - 1):

    current_mid = test.loc[t, "mid"]
    prediction = test.loc[t, "prediction"]
    volatility = test.loc[t, "volatility"]

    # Regression estimates the next log return.
    # Convert it approximately into a price change.
    predicted_price_change = current_mid * prediction

    # Shift quotes toward predicted fair value.
    # Inventory skew encourages reduction of existing positions.
    reservation_price = (
        current_mid
        + ALPHA_STRENGTH * predicted_price_change
        - INVENTORY_SKEW * inventory
    )

    # Widen the spread when volatility is high
    half_spread = BASE_HALF_SPREAD + 0.5 * current_mid * volatility

    bid = reservation_price - half_spread
    ask = reservation_price + half_spread

    # Toy incoming order flow:
    # positive imbalance -> more buyer-initiated orders
    imbalance = test.loc[t, "imbalance"]
    buy_probability = 1 / (1 + np.exp(-imbalance))

    incoming_buy = np.random.random() < buy_probability

    # Simplified fill model.
    # Trades are more likely at tighter quotes.
    fill_probability = np.exp(
        -half_spread / max(current_mid * volatility, 1e-8)
    )
    fill_probability = np.clip(fill_probability, 0.05, 0.95)

    if np.random.random() < fill_probability:

        if incoming_buy and inventory < MAX_INVENTORY:
            # Incoming buyer lifts our ask: we sell
            cash += ask - FEE_PER_UNIT
            inventory -= 1
            fill_count += 1

        elif not incoming_buy and inventory > -MAX_INVENTORY:
            # Incoming seller hits our bid: we buy
            cash -= bid + FEE_PER_UNIT
            inventory += 1
            fill_count += 1

    # Mark inventory to market
    equity = cash + inventory * current_mid

    equity_history.append(equity)
    inventory_history.append(inventory)
    spread_history.append(ask - bid)

test = test.iloc[:-1].copy()
test["strategy_pnl"] = equity_history
test["inventory"] = inventory_history
test["quoted_spread"] = spread_history

# ============================================================
# 5. PERFORMANCE METRICS
# ============================================================

pnl = test["strategy_pnl"]
pnl_changes = pnl.diff().dropna()

total_pnl = pnl.iloc[-1] - pnl.iloc[0]
max_drawdown = (pnl - pnl.cummax()).min()

print("\nMarket-making results")
print("----------------------")
print(f"Total P&L: {total_pnl:.4f}")
print(f"Number of fills: {fill_count}")
print(f"Final inventory: {inventory_history[-1]}")
print(f"Maximum absolute inventory: "
      f"{np.max(np.abs(inventory_history))}")
print(f"Maximum drawdown: {max_drawdown:.4f}")
print(f"Mean quoted spread: {np.mean(spread_history):.4f}")

if pnl_changes.std() > 0:
    print("P&L change Sharpe (unannualised):",
          pnl_changes.mean() / pnl_changes.std())

# ============================================================
# 6. PLOTS
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

axes[0].plot(test["mid"])
axes[0].set_title("Synthetic Mid-Price")
axes[0].set_ylabel("Price")

axes[1].plot(test["strategy_pnl"])
axes[1].set_title("Market-Making Mark-to-Market P&L")
axes[1].set_ylabel("P&L")

axes[2].plot(test["inventory"])
axes[2].set_title("Inventory Over Time")
axes[2].set_ylabel("Units")
axes[2].set_xlabel("Time step")

plt.tight_layout()
plt.show()


# In[ ]:




