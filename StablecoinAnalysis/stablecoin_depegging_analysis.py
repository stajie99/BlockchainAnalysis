import pandas as pd
from datetime import datetime, timezone
import matplotlib.pyplot as plt
# load event data - csv
df_events = pd.read_csv('./ERC20-stablecoins/event_data.csv', encoding='iso-8859-1')
print(df_events.head(10))
print(df_events.info())

# The timestamp e.g. 1660176000 is a Unix timestamp (also called Epoch time),
# convert timestamp column to datetime
df_events['datetime'] = pd.to_datetime(df_events['timestamp'], unit='s')
df_events = df_events.drop('timestamp', axis=1)
print(df_events.head(10))

# load price data - multiple csv files
df_price_dai = pd.read_csv('./ERC20-stablecoins/price_data/dai_price_data.csv')
df_price_dai['stablecoin'] = 'dai'

df_price_pax = pd.read_csv('./ERC20-stablecoins/price_data/pax_price_data.csv')
df_price_pax['stablecoin'] = 'pax'

df_price_usdc = pd.read_csv('./ERC20-stablecoins/price_data/usdc_price_data.csv')
df_price_usdc['stablecoin'] = 'usdc'

df_price_usdt = pd.read_csv('./ERC20-stablecoins/price_data/usdt_price_data.csv')
df_price_usdt['stablecoin'] = 'usdt'

df_price_ustc = pd.read_csv('./ERC20-stablecoins/price_data/ustc_price_data.csv')
df_price_ustc['stablecoin'] = 'ustc'

df_price_wluna = pd.read_csv('./ERC20-stablecoins/price_data/wluna_price_data.csv')
df_price_wluna['stablecoin'] = 'wluna'

# stack vertically
df_price = pd.concat([df_price_dai, df_price_pax, df_price_usdt, df_price_usdc, df_price_ustc, df_price_wluna], axis=0)
print(df_price)
df_price['datetime'] = pd.to_datetime(df_price['timestamp'], unit='s')
df_price = df_price.drop('timestamp', axis=1)
print(df_price)


df_price = df_price.sort_values('datetime') # ensure chronological order

# Multiple stablecoins comparison
plt.figure(figsize=(12, 6))

for coin in df_price['stablecoin'].unique():
    coin_data = df_price[df_price['stablecoin'] == coin]
    plt.plot(coin_data['datetime'], coin_data['close'], label=coin.upper())

plt.title('Daily Close Prices of Stablecoins and WLUNA')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.yscale('log') # Use log scale to visualize both stable coins and the collapse of WLUNA/USTC
# plt.ylim(1e-5, 150) # Set reasonable limits for log scale
plt.show()


# # Subplots for each stablecoin
# stablecoins = df_price['stablecoin'].unique()
# fig, axes = plt.subplots(len(stablecoins), 1, figsize=(12, 4*len(stablecoins)))

# if len(stablecoins) == 1:
#     axes = [axes]

# for i, coin in enumerate(stablecoins):
#     coin_data = df_price[df_price['stablecoin'] == coin]
#     axes[i].plot(coin_data['datetime'], coin_data['close'])
#     axes[i].set_title(f'{coin.upper()} Price')
#     axes[i].set_ylabel('Price')
#     axes[i].grid(True)
#     # axes[i].tick_params(axis='x', rotation=45)

# plt.xlabel('Date')
# plt.tight_layout()
# plt.show()

# Multiple stablecoins (with price ranges) comparison 
plt.figure(figsize=(12, 6))

for coin in df_price['stablecoin'].unique():
    coin_data = df_price[df_price['stablecoin'] == coin]
    plt.fill_between(coin_data['datetime'], 
                    coin_data['low'], 
                    coin_data['high'], 
                    alpha=0.3, label=f'{coin} range')
    plt.plot(coin_data['datetime'], coin_data['close'], 
            label=f'{coin} close', linewidth=2)

plt.title('Daily Prices of Stablecoins and WLUNA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.yscale('log')
plt.show()


# Volatility and Risk Analysis (Daily Range)
# Calculate Daily Range (High - Low) for each coin

plt.figure(figsize=(12, 6))

for coin in df_price['stablecoin'].unique():
    coin_data = df_price[df_price['stablecoin'] == coin]
    plt.plot(coin_data['datetime'], coin_data['high'] - coin_data['low'], 
            label=f'{coin.upper()}', linewidth=1)

plt.title('Daily Intraday Volatility (High - Low)')
plt.xlabel('Date')
plt.ylabel('Daily Price Range (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.yscale('log')
plt.show()

# ###########################################
# Event timeline with vertical lines (overlay on price chart)
fig, ax = plt.subplots(figsize=(14, 8))
import math
# Plot price data first
for coin in df_price['stablecoin'].unique():
    coin_data = df_price[df_price['stablecoin'] == coin]
    ax.plot(coin_data['datetime'], coin_data['close'], label=coin, linewidth=2)

# Add event vertical lines
for i, event in df_events.iterrows():
    # print(i)
    ax.axvline(x=event['datetime'], color='red', linestyle='--', alpha=0.5)
    ax.text(event['datetime'], 
            10**( 2 - i * (2 - (-4)) / df_events.shape[0]),
            f"$\mathbf{{Event\ {i+1}}}$: {event['event'][:30] + "..."}", 
            rotation=0, fontsize=7)

ax.set_title('Prices of Stablecoins and WLUNA with Events')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.yscale('log')
plt.show()



fig, ax = plt.subplots(figsize=(14, 8))
import math
# Plot price data first
for coin in df_price['stablecoin'].unique():
    coin_data = df_price[df_price['stablecoin'] == coin]
    ax.plot(coin_data['datetime'], coin_data['close'], label=coin, linewidth=2)

# Add event vertical lines
for i, event in df_events.iterrows():
    # print(i)
    ax.axvline(x=event['datetime'], color='red', linestyle='--', alpha=0.5)
    ax.text(event['datetime'], 
            10**( 2 - i * (2 - (-4)) / df_events.shape[0]),
            f"$\mathbf{{Event\ {i+1}}}$: {event['type']}", 
            rotation=0, fontsize=7)

ax.set_title('Prices of Stablecoins and WLUNA with Event Sentiment')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.yscale('log')
plt.show()




# Correlation analysis
import seaborn as sns
# 1. Pivot the data to get stablecoins as columns and date as index
pivot_df = df_price.pivot(index='datetime', columns='stablecoin', values='close')

# 2. Calculate the correlation matrix
corr_matrix = pivot_df.corr(method='pearson')

# 3. Generate a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap='coolwarm',
            cbar_kws={'label': 'Pearson Correlation Coefficient (ρ)'},
            linewidths=.5, linecolor='black')
plt.title('Correlation Matrix of Stablecoin/Asset Daily Close Prices')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()





# Random Forest Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Data Preparation ---

# # Load and process USDT price data
# usdt_price_df = pd.read_csv('usdt_price_data.csv')
# usdt_price_df['timestamp'] = pd.to_datetime(usdt_price_df['timestamp'], unit='s').dt.date
# usdt_price_df = usdt_price_df.set_index('timestamp')

# # Load and process event data
# event_df = pd.read_csv('event_data.csv')
# event_df['timestamp'] = pd.to_datetime(event_df['timestamp'], unit='s').dt.date

# Feature Engineering: Count all positive/negative events for all stablecoins
daily_events = df_events.groupby(['datetime', 'stablecoin'])['type'].value_counts().unstack(fill_value=0)
event_features = daily_events.unstack(level='stablecoin').fillna(0)
event_features.columns = [f'{sc}_events_{t}' for t, sc in event_features.columns]

# Merge data and create lagged price feature
merged_df = df_price_usdt.merge(event_features, left_index=True, right_index=True, how='left')
merged_df = merged_df.fillna(0)
merged_df['prev_close'] = merged_df['close'].shift(1)
merged_df.dropna(inplace=True)

# --- Model Training and Evaluation ---

# Define features (X) and target (y)
features = ['prev_close'] + [col for col in merged_df.columns if '_events_' in col]
target = 'close'

X = merged_df[features]
y = merged_df[target]

# Split data: 80% train, 20% test (preserving time order: shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Extract Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
print("Feature Importance:\n", feature_importance)


# visualiza the results
# Create a DataFrame for plotting
results_df = pd.DataFrame({
    'Actual Close': y_test,
    'Predicted Close': y_pred
})

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df['Actual Close'], label='Actual Close Price', color='blue', linewidth=2)
plt.plot(results_df.index, results_df['Predicted Close'], label='Predicted Close Price', color='red', linestyle='--', linewidth=1)
plt.title('Random Forest Model Prediction vs. Actual USDT Close Price (Test Set)')
plt.xlabel('Date')
plt.ylabel('USDT Price (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()



#  Predict wluna price
# --- Data Preparation ---

# Feature Engineering: Count all positive/negative events for all stablecoins
daily_events = df_events.groupby(['datetime', 'stablecoin'])['type'].value_counts().unstack(fill_value=0)
event_features = daily_events.unstack(level='stablecoin').fillna(0)
event_features.columns = [f'{sc}_events_{t}' for t, sc in event_features.columns]

# Merge data and create lagged price feature
merged_df = df_price_wluna.merge(event_features, left_index=True, right_index=True, how='left')
merged_df = merged_df.fillna(0)
merged_df['prev_close'] = merged_df['close'].shift(1)
merged_df.dropna(inplace=True)

# --- Model Training and Evaluation ---
features = ['prev_close'] + [col for col in merged_df.columns if '_events_' in col]
target = 'close'

X = merged_df[features]
y = merged_df[target]

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate metrics and feature importance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
feature_importance = pd.Series(model.feature_importances_, index=features)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
print("Feature Importance:\n", feature_importance)