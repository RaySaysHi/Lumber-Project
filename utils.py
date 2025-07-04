# utils.py
# Helper functions for Exploratory Data Analysis (EDA) on time series data

# --- Imports ---
import pandas as pd
import numpy as np # Needed for histogram bounds
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
# --- End Imports ---


# --- Data Loading and Preparation ---

def load_and_prep_data(filepath):
    """
    Loads time series data from a CSV file, parses the 'DATE' column,
    and sets it as the DataFrame index.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame with DATE as datetime index, or None on error.
    """
    try:
        # Use the imported 'pd' alias here
        df = pd.read_csv(filepath, parse_dates=['DATE'], index_col='DATE')
        print(f"Data loaded successfully from {filepath}")
        print(f"DataFrame shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {df.columns.tolist()}")
        # Optional: Check for missing values
        print(f"\nMissing values per column:\n{df.isnull().sum()}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

# --- Visualization Functions ---

def plot_time_series(df, column_name, title='', xlabel='Date', ylabel=''):
    """
    Plots a single time series column from the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the time series.
        column_name (str): The name of the column to plot.
        title (str, optional): Title for the plot. Defaults to column name.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Date'.
        ylabel (str, optional): Label for the y-axis. Defaults to column name.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        return

    plt.figure(figsize=(15, 5))
    sns.lineplot(data=df, x=df.index, y=column_name)
    plt.title(title if title else f'{column_name} over Time')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else column_name)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_decomposition(df, column_prefix):
    """
    Plots pre-calculated Trend, Seasonal, and Residual components based on
    column prefixes (e.g., 'Trend_Reschedule_Rate').

    Args:
        df (pandas.DataFrame): DataFrame containing the decomposed components.
        column_prefix (str): The base name of the metric (e.g., 'Reschedule_Rate').
    """
    trend_col = f'Trend_{column_prefix}'
    seasonal_col = f'Seasonal_{column_prefix}'
    residual_col = f'Residual_{column_prefix}'

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f'Decomposition for {column_prefix}', y=1.02) # Add overall title

    # Plot Trend
    if trend_col in df.columns:
        sns.lineplot(data=df, x=df.index, y=trend_col, ax=axes[0])
        axes[0].set_title('Trend Component')
        axes[0].set_ylabel('Trend')
        axes[0].grid(True)
    else:
        axes[0].set_title(f'Trend Component for {column_prefix} (Not Found)')

    # Plot Seasonal
    if seasonal_col in df.columns:
        sns.lineplot(data=df, x=df.index, y=seasonal_col, ax=axes[1])
        axes[1].set_title('Seasonal Component')
        axes[1].set_ylabel('Seasonal')
        axes[1].grid(True)
    else:
        axes[1].set_title(f'Seasonal Component for {column_prefix} (Not Found)')

    # Plot Residual
    if residual_col in df.columns:
        sns.lineplot(data=df, x=df.index, y=residual_col, ax=axes[2])
        axes[2].set_title('Residual Component')
        axes[2].set_ylabel('Residual')
        axes[2].grid(True)
    else:
        axes[2].set_title(f'Residual Component for {column_prefix} (Not Found)')

    plt.xlabel('Date') # Add x-label only to the bottom plot
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.show()


def plot_acf_pacf(series, lags=40, title_suffix=''):
    """
    Generates ACF and PACF plots for a given time series.

    Args:
        series (pandas.Series): The time series data (original, differenced, or residuals).
        lags (int, optional): Number of lags to display. Defaults to 40.
        title_suffix (str, optional): Suffix to add to plot titles. Defaults to ''.
    """
    series_clean = series.dropna()
    if series_clean.empty:
        print("Warning: Series is empty after dropping NaNs. Cannot plot ACF/PACF.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    # ACF Plot
    try:
        plot_acf(series_clean, lags=lags, ax=axes[0])
        axes[0].set_title(f'Autocorrelation (ACF) {title_suffix}')
        axes[0].grid(True)
    except Exception as e:
        axes[0].set_title(f'ACF Plot Error: {e}')
        print(f"Error plotting ACF: {e}")

    # PACF Plot
    try:
        # Use 'ywm' method for potentially better estimation with time series
        plot_pacf(series_clean, lags=lags, ax=axes[1], method='ywm')
        axes[1].set_title(f'Partial Autocorrelation (PACF) {title_suffix}')
        axes[1].grid(True)
    except Exception as e:
        axes[1].set_title(f'PACF Plot Error: {e}')
        print(f"Error plotting PACF: {e}")


    plt.tight_layout()
    plt.show()

# --- UPDATED plot_correlation_heatmap function ---
def plot_correlation_heatmap(df, columns=None, method='pearson', title='', figsize=(12, 10)):
    """
    Calculates and displays a heatmap of the correlation matrix for specified columns,
    using either Pearson or Spearman method.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        columns (list, optional): List of column names to include. If None, uses all numeric columns. Defaults to None.
        method (str, optional): Method of correlation ('pearson' or 'spearman'). Defaults to 'pearson'.
        title (str, optional): Title for the heatmap. Defaults to include method name.
        figsize (tuple, optional): Figure size. Defaults to (12, 10).
    """
    plt.figure(figsize=figsize)
    if columns:
        # Ensure only existing columns are selected
        valid_columns = [col for col in columns if col in df.columns]
        if not valid_columns:
             print("Warning: None of the specified columns found for correlation heatmap.")
             return
        data_to_corr = df[valid_columns].select_dtypes(include='number')
    else:
        # Select only numeric columns for correlation
        data_to_corr = df.select_dtypes(include='number')

    if data_to_corr.empty:
        print("Warning: No numeric columns found/selected for correlation heatmap.")
        return

    # Calculate correlation using the specified method
    try:
        corr = data_to_corr.corr(method=method)
    except ValueError:
        print(f"Error: Invalid correlation method '{method}'. Use 'pearson' or 'spearman'.")
        return

    # Generate title if not provided
    if not title:
        title = f'{method.capitalize()} Correlation Matrix'

    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()
# --- END of UPDATED function ---

# --- Statistical Test Functions ---

def run_stationarity_tests(series, series_name=''):
    """
    Performs Augmented Dickey-Fuller (ADF) and KPSS tests for stationarity
    and prints the results.

    Args:
        series (pandas.Series): The time series data to test.
        series_name (str, optional): Name of the series for printing. Defaults to ''.
    """
    series_clean = series.dropna() # Tests require no NaN values
    if series_clean.empty:
        print(f"Warning: Series '{series_name}' is empty after dropping NaNs. Cannot run stationarity tests.")
        return

    print(f'--- Stationarity Tests for: {series_name} ---')

    # ADF Test (H0: Non-stationary)
    try:
        adf_result = adfuller(series_clean)
        print(f'ADF Test Results:')
        print(f'  ADF Statistic: {adf_result[0]:.4f}')
        print(f'  p-value: {adf_result[1]:.4f}')
        print(f'  Lags Used: {adf_result[2]}')
        print(f'  Number of Observations: {adf_result[3]}')
        print(f'  Critical Values:')
        for key, value in adf_result[4].items():
            print(f'\t{key}: {value:.4f}')
        if adf_result[1] <= 0.05:
            print("  Conclusion: Likely Stationary (reject H0 at 5% level)")
        else:
            print("  Conclusion: Likely Non-Stationary (fail to reject H0 at 5% level)")
    except Exception as e:
        print(f"  ADF Test Error: {e}")

    print('\n---')

    # KPSS Test (H0: Stationary around a constant or trend)
    # Test for stationarity around a constant ('c')
    try:
        print(f'KPSS Test Results (stationarity around constant):')
        # Note: nlags='auto' lets statsmodels choose; 'legacy' might be needed for older versions
        kpss_result_c = kpss(series_clean, regression='c', nlags='auto')
        print(f'  KPSS Statistic: {kpss_result_c[0]:.4f}')
        print(f'  p-value: {kpss_result_c[1]:.4f}')
        print(f'  Lags Used: {kpss_result_c[2]}')
        print(f'  Critical Values:')
        for key, value in kpss_result_c[3].items():
            print(f'\t{key}: {value:.4f}')
        if kpss_result_c[1] >= 0.05:
            print("  Conclusion: Likely Stationary around constant (fail to reject H0 at 5% level)")
        else:
            kpss_result_ct = kpss(series_clean, regression='ct', nlags='auto')
            print(f'  KPSS Statistic: {kpss_result_c[0]:.4f}')
            print(f'  p-value: {kpss_result_c[1]:.4f}')
            print(f'  Lags Used: {kpss_result_c[2]}')
            print(f'  Critical Values:')
            if kpss_result_c[1] >= 0.05:
                print("  Conclusion: Likely Stationary around trend (fail to reject H0 at 5% level)")
            else:
                print("  Conclusion: Likely Non-Stationary around trend too (reject H0 at 5% level)")
    except Exception as e:
        print(f"  KPSS Test (constant) Error: {e}")

    # Optional: Test for stationarity around a trend ('ct') if needed
    # try:
    #     print(f'\nKPSS Test Results (stationarity around trend):')
    #     kpss_result_ct = kpss(series_clean, regression='ct', nlags='auto')
    #     print(f'  KPSS Statistic: {kpss_result_ct[0]:.4f}')
    #     print(f'  p-value: {kpss_result_ct[1]:.4f}')
    #     print(f'  Lags Used: {kpss_result_ct[2]}')
    #     print(f'  Critical Values:')
    #     for key, value in kpss_result_ct[3].items():
    #          print(f'\t{key}: {value:.4f}')
    #     if kpss_result_ct[1] >= 0.05:
    #          print("  Conclusion: Likely Stationary around trend (fail to reject H0 at 5% level)")
    #     else:
    #          print("  Conclusion: Likely Non-Stationary around trend (reject H0 at 5% level)")
    # except Exception as e:
    #     print(f"  KPSS Test (trend) Error: {e}")

    print('---------------------------------------\n')

# --- Outlier Detection & Visualization Functions ---

def identify_outliers_iqr(series, series_name='', multiplier=1.5):
    """
    Identifies potential outliers in a Series using the Interquartile Range (IQR) method.

    Args:
        series (pandas.Series): The data series to check for outliers.
        series_name (str, optional): The name of the series for printing warnings. Defaults to ''.
        multiplier (float, optional): The IQR multiplier to define outlier bounds. Defaults to 1.5.

    Returns:
        tuple: A tuple containing:
            - pandas.Index: The index values of the identified outliers.
            - float: The calculated lower bound for outliers.
            - float: The calculated upper bound for outliers.
            - bool: True if IQR was zero, False otherwise.
    """
    if series.empty:
        print(f"Warning: Series '{series_name}' is empty. Cannot identify outliers.")
        return pd.Index([]), np.nan, np.nan, False

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    iqr_zero = False

    # Handle cases where IQR might be zero (e.g., constant data)
    if iqr == 0:
        print(f"Warning: IQR for '{series_name}' is zero. Outlier detection based on IQR might not be meaningful.")
        iqr_zero = True
        # Define bounds based on median or simply use Q1/Q3 which are the same
        lower_bound = q1
        upper_bound = q3
        median_val = series.median()
        outliers = series[series != median_val]
    else:
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]

    print(f"Identified {len(outliers)} potential outliers for '{series_name}' using IQR method (multiplier={multiplier}). Bounds: ({lower_bound:.4f}, {upper_bound:.4f})")

    return outliers.index, lower_bound, upper_bound, iqr_zero

def plot_histogram_with_outliers(series, series_name='', multiplier=1.5, bins=30):
    """
    Plots a histogram of the series and marks the IQR outlier bounds.

    Args:
        series (pandas.Series): The data series to plot.
        series_name (str, optional): Name for the plot title. Defaults to ''.
        multiplier (float, optional): IQR multiplier for bounds. Defaults to 1.5.
        bins (int, optional): Number of bins for the histogram. Defaults to 30.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(series, bins=bins, kde=True)
    plt.title(f'Histogram of {series_name}')

    # Calculate bounds (using the same logic as identify_outliers_iqr)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if iqr > 0: # Only draw lines if IQR is not zero
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        plt.axvline(lower_bound, color='r', linestyle='--', label=f'Lower Bound ({lower_bound:.2f})')
        plt.axvline(upper_bound, color='r', linestyle='--', label=f'Upper Bound ({upper_bound:.2f})')
        plt.legend()
    else:
        plt.axvline(q1, color='orange', linestyle='--', label=f'Q1/Median/Q3 ({q1:.2f})')
        plt.legend()


    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_series_with_outliers(df, column_name, title='', xlabel='Date', ylabel='', multiplier=1.5):
    """
    Plots a time series and highlights outliers identified by the IQR method.

    Args:
        df (pandas.DataFrame): DataFrame containing the time series.
        column_name (str): The name of the column to plot.
        title (str, optional): Title for the plot. Defaults to column name.
        xlabel (str, optional): Label for the x-axis. Defaults to 'Date'.
        ylabel (str, optional): Label for the y-axis. Defaults to column name.
        multiplier (float, optional): IQR multiplier used to identify outliers. Defaults to 1.5.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        return

    # Identify outliers first
    outlier_indices, _, _, _ = identify_outliers_iqr(df[column_name], series_name=column_name, multiplier=multiplier)

    plt.figure(figsize=(15, 5))
    # Plot the main series
    sns.lineplot(data=df, x=df.index, y=column_name, label=column_name, zorder=1)

    # Overlay outliers if any exist
    if not outlier_indices.empty:
        sns.scatterplot(data=df.loc[outlier_indices], x=outlier_indices, y=column_name,
                        color='red', s=50, label='Outliers (IQR)', zorder=2)

    plt.title(title if title else f'{column_name} over Time with Outliers')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else column_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- Example Usage (if run as script) ---
if __name__ == '__main__':
    # This block runs only if the script is executed directly
    print("utils.py script loaded. Contains helper functions for time series analysis.")
    # Example:
    # filepath = 'lumber_delivery_preprocessed.csv'
    # df = load_and_prep_data(filepath)
    # if df is not None:
    #     # plot_histogram_with_outliers(df['Reschedule_Rate'], series_name='Reschedule_Rate')
    #     # plot_series_with_outliers(df, 'Reschedule_Rate', title='Reschedule Rate with Outliers')
    #     corr_cols = ['Reschedule_Rate', 'Truck_Utilization_Efficiency', 'WPU081', 'GASREGW', 'DFF']
    #     plot_correlation_heatmap(df, columns=corr_cols, method='spearman', title='Spearman Correlation') # Example Spearman