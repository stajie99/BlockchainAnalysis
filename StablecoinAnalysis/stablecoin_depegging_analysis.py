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
# # 1. Calculate Returns (log returns are better for correlation analysis)
# returns_df = np.log(pivot_df / pivot_df.shift(1)).dropna()
# print("\nDaily Log Returns:")
# print(returns_df.head())
# # returns_df = returns_df.drop('wluna', axis=1)

# # Step : Feature Engineering - adding technical indicators
# pivot_df['MA_20'] = pivot_df['Close'].rolling(window=20).mean()
# pivot_df['MA_50'] = pivot_df['Close'].rolling(window=50).mean()
# pivot_df['Volatility'] = pivot_df['Close'].rolling(window=20).std()
# pivot_df.dropna(inplace=True)
# # Step 4: Data Preprocessing - Scaling the data
# features = pivot_df[['Close', 'MA_20', 'MA_50', 'Volatility']]
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)
# # Step 5: Apply PCA
# pca = PCA(n_components=2)  # Retaining 2 components
# principal_components = pca.fit_transform(scaled_features)
# # Step 6: Visualizing Explained Variance
# explained_variance = pca.explained_variance_ratio_
# plt.bar(range(len(explained_variance)), explained_variance)
# plt.title('Explained Variance by Principal Components')
# plt.show()
# # Step 7: Building the Predictive Model
# X = principal_components
# y = pivot_df['Close']  # Using 'Close' price as the target
# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Using Linear Regression for simplicity
# model = LinearRegression()
# model.fit(X_train, y_train)
# # Step 8: Evaluating the Model
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# # Step 9: Performance Comparison - Model without PCA
# model_no_pca = LinearRegression()
# model_no_pca.fit(X_train, y_train)
# y_pred_no_pca = model_no_pca.predict(X_test)
# mse_no_pca = mean_squared_error(y_test, y_pred_no_pca)
# print(f'MSE without PCA: {mse_no_pca}')
# # Step 10: Compare the Results
# print(f'Performance with PCA: {mse}')
# print(f'Performance without PCA: {mse_no_pca}')

def negative_news_ratio(df_events, returns_df, window = 7):
    # Select key dates
    start_date = returns_df.index[window - 1]
    end_date = returns_df.index[-1]
    
    # Create weekly intervals
    dates = pd.date_range(start=start_date , end=end_date)
    nega_ratio = []
    
    for current_date in dates:
        # Get events in the past 7 days (inclusive)
        window_start = current_date - pd.Timedelta(days=window-1)  # 7-day window
        window_events = df_events[(df_events['datetime'] >= window_start) & 
                          (df_events['datetime'] <= current_date)]
        
        total = len(window_events)
        negative = (window_events['type'] == 'negative').sum()
        ratio = negative / total if total > 0 else 0
        
        nega_ratio.append({
            'date': current_date,
            f'total_events_{window}d': total,
            f'negative_events_{window}d': negative,
            f'negative_ratio_{window}d': ratio
        })
    # Convert to DataFrame
    nega_ratio_df = pd.DataFrame(nega_ratio, columns=['date', f'total_events_{window}d',
                                                       f'negative_events_{window}d', 
                                                       f'negative_ratio_{window}d'])

    plt.figure(figsize=(12, 6))
    plt.plot(nega_ratio_df['date'], nega_ratio_df[f'negative_ratio_{window}d'], 
             linewidth=2, label=f'negative_ratio_{window}d')
    plt.title('Negative News Ratio over Time')
    plt.xlabel('Date')
    plt.legend()
    # plt.grid(True)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return nega_ratio_df

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def pca_timeline(df, n_components = 3):
    """
    Create a series of correlation heatmaps/matrice over time.
    window:     the look-back period when calculating correlation matrix of stablecoin returns;
                the longer, the slower to reflect latest market move; the shorter, the more fluctuation in correlation structure
    frequency_days:     the frequency we report the changes in two adjacent correlation matrice.

    Specialized PCA analysis for stablecoin correlations.
    """
    # Standardize unless otherwise stated
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)  # Zero mean, unit variance
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    # Fit PCA
    X_pca = pca.fit_transform(df_scaled)
    
    # Get results
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_  # Shape: (n_components, n_assets)
    
    # Create results dictionary
    results = {
        'pca': pca,
        'X_pca': X_pca,
        'components': components,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'asset_names': df.columns.tolist(),
        'dates': df.index,
    }
    
    # Create focused visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. PC time series (market factor)
    axes[0, 0].plot(results['dates'], results['X_pca'][:, 0], 
                   linewidth=2, color='darkblue', label='PC1 (Market Factor)')
    axes[0, 0].fill_between(results['dates'], results['X_pca'][:, 0], 
                           alpha=0.3, color='blue')
    axes[0, 0].set_title('Market Factor (PC1) Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('PC1 Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. PC loadings heatmap
    loadings = results['components'][:min(3, n_components), :]
    asset_names = results['asset_names']
    
    im = axes[0, 1].imshow(loadings, cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('Stablecoin Loadings on Principal Components')
    axes[0, 1].set_xlabel('Stablecoins')
    axes[0, 1].set_ylabel('Principal Components')
    axes[0, 1].set_xticks(range(len(asset_names)))
    axes[0, 1].set_yticks(range(min(3, n_components)))
    axes[0, 1].set_xticklabels(asset_names, rotation=45, ha='right')
    axes[0, 1].set_yticklabels([f'PC{i+1}' for i in range(min(3, n_components))])
    plt.colorbar(im, ax=axes[0, 1])
    
    # 3. PC1 loadings bar chart
    pc1_loadings = results['components'][0, :]
    colors = ['green' if x > 0 else 'red' for x in pc1_loadings]
    
    axes[1, 0].barh(asset_names, pc1_loadings, color=colors, alpha=0.7)
    axes[1, 0].axvline(x=0, color='black', linewidth=1)
    axes[1, 0].set_title('PC1 Loadings (Market Factor Exposure)')
    axes[1, 0].set_xlabel('Loading Value')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Add values to bars
    for i, (name, loading) in enumerate(zip(asset_names, pc1_loadings)):
        axes[1, 0].text(loading + (0.01 if loading > 0 else -0.01), i,
                       f'{loading:.3f}', va='center',
                       ha='left' if loading > 0 else 'right', fontsize=9)
    
    # 4. Cumulative variance explained
    explained_ratio = results['explained_variance_ratio']
    cumulative = np.cumsum(explained_ratio)
    
    axes[1, 1].bar(range(1, len(explained_ratio) + 1), explained_ratio, 
                  alpha=0.6, color='steelblue', label='Individual')
    axes[1, 1].plot(range(1, len(cumulative) + 1), cumulative, 
                   'ro-', linewidth=2, markersize=6, label='Cumulative')
    axes[1, 1].set_title('Variance Explained by Principal Components')
    axes[1, 1].set_xlabel('Principal Component')
    axes[1, 1].set_ylabel('Variance Explained')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1.1)
    
    # Add annotations
    for i, (indiv, cum) in enumerate(zip(explained_ratio, cumulative), 1):
        axes[1, 1].text(i, indiv + 0.02, f'{indiv*100:.0f}%', 
                       ha='center', fontsize=8)
        if i == len(explained_ratio):
            axes[1, 1].text(i, cum + 0.02, f'{cum*100:.0f}%', 
                           ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Stablecoin PCA Analysis ({df.shape[0]} days)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Interpretation
    print("\n" + "="*70)
    print("STABLECOIN PCA INTERPRETATION")
    print("="*70)
    
    loadings_df = pd.DataFrame(
        results['components'].T,
        index=asset_names,
        columns=[f'PC{i+1}' for i in range(results['components'].shape[0])]
    )
    
    print(f"\nFirst {n_components} PCs explain {explained_ratio[:n_components].sum()*100:.1f}% of variance")
    
    print(f"\nPC1 (Market Factor) Loadings:")
    print(loadings_df['PC1'].sort_values(ascending=False).round(4))
    
    print(f"\nAssets most correlated with market (|PC1| > 0.3):")
    high_corr = loadings_df['PC1'][abs(loadings_df['PC1']) > 0.3]
    if len(high_corr) > 0:
        print(high_corr.sort_values(ascending=False).round(4))
    else:
        print("None")
    
    return {
        'loadings_df': loadings_df,
        'pc_series': pd.DataFrame(results['X_pca'], index=results['dates'],
                                 columns=[f'PC{i+1}' for i in range(results['X_pca'].shape[1])]),
        'explained_variance_ratio': explained_ratio
    }

nega_ratio_df = negative_news_ratio(df_events, returns_df, window=7)
results_pca = pca_timeline(df=pivot_df['2022-04-03':'2022-06-30'] , n_components = 3)



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