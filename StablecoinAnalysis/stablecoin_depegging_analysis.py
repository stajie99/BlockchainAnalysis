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

plt.title('Stablecoin Price Ranges')
plt.xlabel('Date')
plt.ylabel('Price')
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
    ax.axvline(x=event['datetime'], color='red', linestyle='--', alpha=0.7)
    ax.text(event['datetime'], 
            (ax.get_ylim()[1] - i * (ax.get_ylim()[1] - ax.get_ylim()[0]) / df_events.shape[0]),
            f"Event {i+1}", 
            rotation=90, fontsize=8)

ax.set_title('Stablecoin Prices with Events')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.yscale('log')
plt.show()




# Event annotations on price chart
# from adjustText import adjust_text
fig, ax = plt.subplots(figsize=(14, 8))

# Plot price data
for coin in df_price['stablecoin'].unique():
    coin_data = df_price[df_price['stablecoin'] == coin]
    ax.plot(coin_data['datetime'], coin_data['close'], label=coin, linewidth=2)

# Add event annotations with text
for i, event in df_events.iterrows():
    # Find price at event date for positioning
    event_date = event['datetime']
    # Get approximate y position (you might need to adjust this)
    y_pos = ax.get_ylim()[1] - 0.001 * (i % 3)  # Stagger labels
    
    ax.annotate(event['event'][:30] + "...",  # Truncate long text
               xy=(event_date, y_pos),
               xytext=(10, 30), textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
               fontsize=8, ha='left', rotation = 30)

ax.set_title('Stablecoin Prices with Event Annotations')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.ylim(0.7, 1.2)
plt.show()

# Event table + Price chart combination

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 1, height_ratios=[3, 1], figure=fig)

# Top subplot: Price chart
ax1 = fig.add_subplot(gs[0])
for coin in df_price['stablecoin'].unique():
    coin_data = df_price[df_price['stablecoin'] == coin]
    ax1.plot(coin_data['datetime'], coin_data['close'], label=coin, linewidth=2)

# Add event vertical lines
for i, event in df_events.iterrows():
    ax1.axvline(x=event['datetime'], color='red', linestyle='--', alpha=0.5)

ax1.set_title('Stablecoin Prices with Events')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom subplot: Event table
ax2 = fig.add_subplot(gs[1])
ax2.axis('tight')
ax2.axis('off')

# Create table data
table_data = []
for i, event in df_events.iterrows():
    table_data.append([event['datetime'].strftime('%Y-%m-%d'), event['event'][:50] + "..."])

table = ax2.table(cellText=table_data,
                 colLabels=['Date', 'Event Description'],
                 loc='center',
                 cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

plt.tight_layout()
plt.show()