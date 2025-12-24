import pandas as pd
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import seaborn as sns

############# Part 1. Data Loading
#############
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

df_price = df_price.sort_values(by=['datetime', 'stablecoin']) # ensure chronological order

# ############# Part 2. Simple Data Analysis
# #############
# print(df_events['type'].value_counts())
# print(df_events[['stablecoin','type']].value_counts())

# # Convert to unstacked df
# plot_data = df_events[['stablecoin','type']].value_counts().unstack(fill_value=0)
# # Create grouped bar chart
# # plt.figure(figsize=(12, 6))
# plot_data.plot(kind='bar', color=['red', 'green'])
# plt.title('Grouped Bar Chart', fontsize=14)
# plt.xlabel('Stablecoin Type')
# plt.ylabel('Count')
# plt.legend(title='Sentiment')
# plt.tick_params(axis='x', rotation=45)
# plt.show()

# # Time Series Plot (Line Chart) of event frequency (e.g., events per week or month) 
# df_events['week'] = df_events['datetime'].dt.to_period('W').dt.start_time

# # Method : Using Seaborn for a cleaner look
# plt.figure(figsize=(14, 8))
# sns.lineplot(data=df_events.groupby(['week', 'type']).size().reset_index(name='count'),
#              x='week', y='count', hue='type',
#              palette=['green', 'red'],
#              marker='o', linewidth=2.5, markersize=10)

# plt.title('Event Frequency Over Time (Aggragated by Week)', fontsize=16, fontweight='bold', pad=20)
# plt.xlabel('Week (Start Date)', fontsize=12)
# plt.ylabel('Number of Events', fontsize=12)
# plt.legend(title='Event Type')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # Multiple stablecoins comparison
# plt.figure(figsize=(12, 6))

# for coin in df_price['stablecoin'].unique():
#     coin_data = df_price[df_price['stablecoin'] == coin]
#     plt.plot(coin_data['datetime'], coin_data['close'], label=coin.upper())

# plt.title('Daily close Prices of Stablecoins and WLUNA')
# plt.xlabel('Date')
# plt.ylabel('close Price (USD)')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.yscale('log') # Use log scale to visualize both stable coins and the collapse of WLUNA/USTC
# # plt.ylim(1e-5, 150) # Set reasonable limits for log scale
# plt.show()


# # # Subplots for each stablecoin
# # stablecoins = df_price['stablecoin'].unique()
# # fig, axes = plt.subplots(len(stablecoins), 1, figsize=(12, 4*len(stablecoins)))

# # if len(stablecoins) == 1:
# #     axes = [axes]

# # for i, coin in enumerate(stablecoins):
# #     coin_data = df_price[df_price['stablecoin'] == coin]
# #     axes[i].plot(coin_data['datetime'], coin_data['close'])
# #     axes[i].set_title(f'{coin.upper()} Price')
# #     axes[i].set_ylabel('Price')
# #     axes[i].grid(True)
# #     # axes[i].tick_params(axis='x', rotation=45)

# # plt.xlabel('Date')
# # plt.tight_layout()
# # plt.show()

# # Multiple stablecoins (with price ranges) comparison 
# plt.figure(figsize=(12, 6))

# for coin in df_price['stablecoin'].unique():
#     coin_data = df_price[df_price['stablecoin'] == coin]
#     plt.fill_between(coin_data['datetime'], 
#                     coin_data['low'], 
#                     coin_data['high'], 
#                     alpha=0.3, label=f'{coin} range')
#     plt.plot(coin_data['datetime'], coin_data['close'], 
#             label=f'{coin} close', linewidth=2)

# plt.title('Daily Prices of Stablecoins and WLUNA')
# plt.xlabel('Date')
# plt.ylabel('Price (USD)')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.yscale('log')
# plt.show()


# # Volatility and Risk Analysis (Daily Range)
# # Calculate Daily Range (High - Low) for each coin

# plt.figure(figsize=(12, 6))

# for coin in df_price['stablecoin'].unique():
#     coin_data = df_price[df_price['stablecoin'] == coin]
#     plt.plot(coin_data['datetime'], coin_data['high'] - coin_data['low'], 
#             label=f'{coin.upper()}', linewidth=1)

# plt.title('Daily Intraday Volatility (High - Low)')
# plt.xlabel('Date')
# plt.ylabel('Daily Price Range (USD)')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.yscale('log')
# plt.show()

# # ###########################################
# # Event timeline with vertical lines (overlay on price chart)
# fig, ax = plt.subplots(figsize=(14, 8))
# import math
# # Plot price data first
# for coin in df_price['stablecoin'].unique():
#     coin_data = df_price[df_price['stablecoin'] == coin]
#     ax.plot(coin_data['datetime'], coin_data['close'], label=coin, linewidth=2)

# # Add event vertical lines
# for i, event in df_events.iterrows():
#     # print(i)
#     ax.axvline(x=event['datetime'], color='red', linestyle='--', alpha=0.5)
#     ax.text(event['datetime'], 
#             10**( 2 - i * (2 - (-4)) / df_events.shape[0]),
#             f"$\mathbf{{Event\ {i+1}}}$: {event['event'][:30] + "..."}", 
#             rotation=0, fontsize=7)

# ax.set_title('Prices of Stablecoins and WLUNA with Events')
# ax.set_xlabel('Date')
# ax.set_ylabel('Price (USD)')
# ax.legend()
# ax.grid(True, alpha=0.3)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.yscale('log')
# plt.show()



# fig, ax = plt.subplots(figsize=(14, 8))
# import math
# # Plot price data first
# for coin in df_price['stablecoin'].unique():
#     coin_data = df_price[df_price['stablecoin'] == coin]
#     ax.plot(coin_data['datetime'], coin_data['close'], label=coin, linewidth=2)

# # Add event vertical lines
# for i, event in df_events.iterrows():
#     # print(i)
#     ax.axvline(x=event['datetime'], color='red', linestyle='--', alpha=0.5)
#     ax.text(event['datetime'], 
#             10**( 2 - i * (2 - (-4)) / df_events.shape[0]),
#             f"$\mathbf{{Event\ {i+1}}}$: {event['type']}", 
#             rotation=0, fontsize=7)

# ax.set_title('Prices of Stablecoins and WLUNA with Event Sentiment')
# ax.set_xlabel('Date')
# ax.set_ylabel('Price (USD)')
# ax.legend()
# ax.grid(True, alpha=0.3)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.yscale('log')
# plt.show()




# Correlation analysis
import seaborn as sns
# 1. Pivot the data to get stablecoins as columns and date as index
pivot_df = df_price.pivot(index='datetime', columns='stablecoin', values='close')
# 1. Calculate Returns (log returns are better for correlation analysis)
returns_df = np.log(pivot_df / pivot_df.shift(1)).dropna()
print("\nDaily Log Returns:")
print(returns_df.head())
returns_df = returns_df.drop('wluna', axis=1)
# # 2. Calculate the correlation matrix
# corr_matrix = returns_df.corr(method='pearson')

def correlation_heatmap_change_timeline(returns_df, frequency_days=1, window=7):
    """
    Create a series of correlation heatmaps/matrice over time.
    window:     the look-back period when calculating correlation matrix of stablecoin returns;
                the longer, the slower to reflect latest market move; the shorter, the more fluctuation in correlation structure
    frequency_days:     the frequency we report the changes in two adjacent correlation matrice.
    """
    # Select key dates
    start_date = returns_df.index[window - 1]
    end_date = returns_df.index[-1]
    
    # Create weekly intervals
    dates = pd.date_range(start=start_date , end=end_date, freq=f'{frequency_days}D')
    
    
    corr_change = []
    # corr_ = np.zeros((len(returns_df.columns), len(returns_df.columns)))   
    corr_ = returns_df.iloc[0:window].corr(method='pearson').fillna(0)

    for i, date in enumerate(dates):  
        print(date)
        if date in returns_df.index:
            idx = returns_df.index.get_loc(date)
            start_idx = max(0, idx - window)
            rolling_wd_data = returns_df.iloc[start_idx:idx]

            corr_matrix = rolling_wd_data.corr(method='pearson').fillna(0) # Pearson correlation matrix
            diff_corr = corr_matrix #- corr_ 
            corr_change.append(round(np.linalg.norm(diff_corr, 'fro'), 4)) # magnititude change of correlation matrix - Frobenius norm
            # Convert np.float64 to regular floats if needed
            corr_change = [float(x) for x in corr_change] 
            # corr_ = corr_matrix
            
            # Plot heatmap
            # fig, axes = plt.subplots(1, len(dates), figsize=(20, 5))
            fig, axes = plt.subplots(ncols=3, figsize=(20, 5))
            sns.heatmap(corr_matrix, 
                       ax=axes[i],
                       annot=True, 
                       fmt='.2f',
                       cmap='RdBu_r',
                       center=0,
                       vmin=-1, vmax=1,
                       square=True,
                       cbar_kws={'shrink': 0.8})
            
            axes[i].set_title(f'{rolling_wd_data.index[0].date()} to {rolling_wd_data.index[-1].date()}')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].tick_params(axis='y', rotation=0)
    
    plt.suptitle(f'Evolution of Stablecoin Correlations ({window}-Day Rolling)', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()
    
    corr_change = pd.DataFrame(corr_change, index=dates, columns=['corr_change'])
    # print(corr_change)
    # print(dates)
    

    plt.figure(figsize=(12, 6))
    plt.plot(corr_change.index, corr_change['corr_change'], 
             linewidth=2, label='Correlation Change')
    plt.title('Magnitude of Change in Correlation over Time')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Create correlation timeline
correlation_heatmap_change_timeline(returns_df, frequency_days=1, window=7)





# 3. Generate a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap='coolwarm',
            cbar_kws={'label': 'Pearson Correlation Coefficient (ρ)'},
            linewidths=.5, linecolor='black')
plt.title('Correlation Matrix of Stablecoin/Asset Daily close Prices')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
######################
############################### Plots Done




# Random Forest Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Data Preparation ---
# Feature Engineering 1: Count all positive/negative events for all stablecoins
daily_events = df_events.groupby(['datetime', 'stablecoin'])['type'].value_counts().unstack(fill_value=0)
event_features = daily_events.unstack(level='stablecoin').fillna(0)
event_features.columns = [f'{sc}_events_{ty}' for ty, sc in event_features.columns]

# # Merge with event data
# merged_df = df_price.merge(event_features, on='datetime', how='left')
# merged_df = merged_df.fillna(0)

# Feature Engineering 2: Create lagged price feature. Pivot with the constant group : datatime
df_price['prev_close'] = df_price.groupby('stablecoin')['close'].shift(1)
df_price.dropna(inplace=True)
# 
lagged_price_features = df_price.pivot(index='datetime', columns='stablecoin', values='prev_close')
# lagged_price_features = lagged_price_features.reset_index()
lagged_price_features.columns = [f'{sc}_close_(t-1)' for sc in lagged_price_features.columns]

# merged_df = merged_df.drop(['stablecoin', 'prev_close'], axis=1).set_index('datetime')  # Set datetime as new index

# lagged_price_features = pd.concat([merged_df['datetime'], lagged_price_features], axis=1)
# print(lagged_price_features)

merged_df = lagged_price_features.merge(event_features, on='datetime', how='left')
merged_df = merged_df.fillna(0)
merged_df = df_price.loc[df_price['stablecoin'] == 'ustc', ['datetime', 'close']].merge(merged_df, on='datetime', how='left')
merged_df = merged_df.set_index('datetime')
merged_df.to_csv('merged_data.csv')
df_price.to_csv('price_data.csv')
# --- Model Training and Evaluation ---

# Define features (X) and target (y)
# features = ['prev_close'] + [col for col in merged_df.columns if '_events_' in col or '(t-1)' in col]
features = [col for col in merged_df.columns if '_events_' in col or '(t-1)' in col]
target = 'close'

X = merged_df[features]
y = merged_df[target]

# merged_df[['datetime','stablecoin']]
# Split data: 80% train, 20% test (preserving time order: shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


########## Feature-to-Sample Ratio
n_samples = X_train.shape[0]
n_features = X_train.shape[1]
ratio = n_samples / n_features
    
print(f"\n📊 DATA DIMENSIONALITY CHECK:")
print(f"Samples: {n_samples}")
print(f"Features: {n_features}")
print(f"Samples/Feature ratio: {ratio:.1f}")
####################
# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1, 
                                 max_depth=5,               # Shallower trees
                                 min_samples_split=10      # Require more samples to split
                                #  min_samples_leaf=5,        # Larger leaf nodes
                                #  max_features='sqrt',       # Use fewer features per split
                                #  bootstrap=True             # Use bootstrap sampling
                                 ))
        ])
# Fit pipeline: train Random Forest Regressor
pipeline.fit(X_train, y_train)
# Predictions are automatically in original scale
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Extract Feature Importance
feature_importance = pd.Series(pipeline.named_steps['rf'].feature_importances_, index=features).sort_values(ascending=False)


print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
print("Feature Importance:\n", feature_importance)


# visualiza the results
# Create a DataFrame for plotting
results_df = pd.DataFrame({
    'Actual close': y_test,
    'Predicted close': y_pred
})
results_df.to_csv('prediction_results_data.csv')
# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df['Actual close'], label='USTC Actual close Price', color='blue', linewidth=2)
plt.plot(results_df.index, results_df['Predicted close'], label='USTC Predicted close Price', color='red', linestyle='--', linewidth=1)
plt.title('Random Forest Model Prediction vs. Actual USTC close Price (Test Set)')
plt.xlabel('Date')
plt.ylabel('USTC Price (USD)')
plt.legend()
# plt.xticks(rotation=45)
plt.grid(True, linestyle=':', alpha=0.6)
# plt.tight_layout()
plt.savefig('ustc_prediction_vs_actual.png')
plt.show()


################################ for internal analysis


# Make predictions for both training and testing
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)
# Calculate metrics for training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
# Calculate metrics for testing set
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print metrics comparison
print("MODEL EVALUATION METRICS")
print(f"{'Metric':<20} {'Training':<15} {'Testing':<15}")
print(f"{'R-squared':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
print(f"{'MSE':<20} {train_mse:<15.4f} {test_mse:<15.4f}")

# Check for overfitting
overfit_r2 = train_r2 - test_r2
print(f"\nOverfitting indicator (R² diff): {overfit_r2:.4f}")
if overfit_r2 > 0.1:
    print("⚠️  Warning: Possible overfitting (training R² much higher than test R²)")


#########################################################
#################  Time Series model fitting
import statsmodels.api as sm
# Add constant (intercept) column
X_train_with_const = sm.add_constant(X_train)
X_test_with_const = sm.add_constant(X_test)

# Fit OLS model
ols_model = sm.OLS(y_train, X_train_with_const)  #Initializes the model structure.
ols_fit = ols_model.fit()
# Get full summary with p-values
print(ols_fit.summary())
print(ols_fit.pvalues)
# Predictions
del y_pred
y_pred = ols_fit.predict(X_test_with_const)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# visualiza the results
# Create a DataFrame for plotting

results_df = pd.DataFrame({
    'Actual close': y_test,
    'Predicted close': y_pred
})
results_df.to_csv('prediction_results_data.csv')
# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df['Actual close'], label='USTC Actual close Price', color='blue', linewidth=2)
plt.plot(results_df.index, results_df['Predicted close'], label='USTC Predicted close Price', color='red', linestyle='--', linewidth=1)
plt.title('Time Series Model Prediction vs. Actual USTC close Price (Test Set)')
plt.xlabel('Date')
plt.ylabel('USTC Price (USD)')
plt.legend()
# plt.xticks(rotation=45)
plt.grid(True, linestyle=':', alpha=0.6)
# plt.tight_layout()
plt.savefig('ts_ustc_prediction_vs_actual.png')
plt.show()