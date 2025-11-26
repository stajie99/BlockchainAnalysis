import pandas as pd
from datetime import datetime, timezone
import matplotlib.pyplot as plt
# load event data - csv
df_events = pd.read_csv('./ERC20-stablecoins/event_data.csv', encoding='iso-8859-1')
print(df_events.head(10))

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

df_price_wluna = pd.read_csv('./ERC20-stablecoins/price_data/pax_price_data.csv')
df_price_wluna['stablecoin'] = 'wluna'

# stack vertically
df_price = pd.concat([df_price_dai, df_price_pax, df_price_usdt, df_price_usdc, df_price_ustc, df_price_wluna], axis=0)
print(df_price)
df_price['datetime'] = pd.to_datetime(df_price['timestamp'], unit='s')
df_price = df_price.drop('timestamp', axis=1)
print(df_price)



# Simplest method - plots all numeric columns
plt.figure(figsize=(12, 6))
df_price.plot(title='All Time Series', figsize=(12, 6))
plt.ylabel('Value')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




