import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np # Import numpy

# Suppress specific pandas warnings (optional)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# --- Configuration ---
DATA_FILE = 'portfolio_transactions.csv'
DATE_FORMAT = '%Y-%m-%d' # Consistent date format

# --- File Handling (load_transactions, save_transactions - Keep as is) ---
# ... (Keep the load_transactions and save_transactions functions as before) ...
def load_transactions(filename=DATA_FILE):
    """Loads transactions from the CSV file into a pandas DataFrame."""
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, parse_dates=['Date'])
            # Ensure correct data types after loading
            df['Date'] = pd.to_datetime(df['Date']) # Convert to datetime objects
            df['Quantity'] = pd.to_numeric(df['Quantity'])
            df['Price'] = pd.to_numeric(df['Price'])
            df['Account'] = df['Account'].astype(str)
            df['Ticker'] = df['Ticker'].astype(str)
            df['Action'] = df['Action'].astype(str)
            print(f"Loaded {len(df)} transactions from {filename}")
            return df
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            print("Starting with an empty portfolio.")
            # Fallback to empty DataFrame if load fails
            return pd.DataFrame(columns=['Account', 'Ticker', 'Action', 'Quantity', 'Price', 'Date'])
    else:
        print(f"{filename} not found. Starting with an empty portfolio.")
        return pd.DataFrame(columns=['Account', 'Ticker', 'Action', 'Quantity', 'Price', 'Date'])

def save_transactions(df, filename=DATA_FILE):
    """Saves the DataFrame to the CSV file."""
    try:
        # Ensure Date column is in the correct string format for CSV saving if needed
        df_copy = df.copy()
        df_copy['Date'] = df_copy['Date'].dt.strftime(DATE_FORMAT)
        df_copy.to_csv(filename, index=False)
        print(f"Transactions saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving transactions to {filename}: {e}")

# --- Core Logic (add_transaction, get_holdings, display_holdings_report - Keep as is) ---
# ... (Keep the add_transaction, get_holdings, display_holdings_report functions as before) ...
def add_transaction(df):
    """Prompts user for transaction details and adds it to the DataFrame."""
    print("\n--- Add New Transaction ---")
    while True:
        account = input("Enter Account Name: ").strip()
        if account: break
        else: print("Account name cannot be empty.")

    while True:
        ticker = input("Enter Ticker Symbol (e.g., RELIANCE.NS, INFY.NS): ").strip().upper()
        if ticker: break
        else: print("Ticker symbol cannot be empty.")

    while True:
        action = input("Enter Action (BUY/SELL): ").strip().upper()
        if action in ['BUY', 'SELL']: break
        else: print("Invalid action. Please enter BUY or SELL.")

    while True:
        try:
            quantity = float(input("Enter Quantity: ").strip())
            if quantity > 0: break
            else: print("Quantity must be positive.")
        except ValueError:
            print("Invalid quantity. Please enter a number.")

    while True:
        try:
            price = float(input("Enter Price per Share: ").strip())
            if price >= 0: break
            else: print("Price cannot be negative.")
        except ValueError:
            print("Invalid price. Please enter a number.")

    while True:
        date_str = input(f"Enter Transaction Date ({DATE_FORMAT.replace('%', '').upper()}), leave blank for today: ").strip()
        try:
            if not date_str:
                # Use date only, setting time to start of day for comparisons
                trans_date = pd.Timestamp(datetime.now().date())
            else:
                # Use date only, setting time to start of day for comparisons
                 trans_date = pd.Timestamp(datetime.strptime(date_str, DATE_FORMAT).date())
            break
        except ValueError:
            print(f"Invalid date format. Please use {DATE_FORMAT.replace('%', '').upper()}.")

    new_transaction = pd.DataFrame([{
        'Account': account,
        'Ticker': ticker,
        'Action': action,
        'Quantity': quantity,
        'Price': price,
        'Date': trans_date # Store as Timestamp
    }])

    # Ensure Date column in main df is also Timestamp for proper concatenation
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
    df = pd.concat([df, new_transaction], ignore_index=True)
    print("Transaction added successfully.")
    return df

def get_holdings(df, target_account=None, target_date=None):
    """
    Calculates holdings (Quantity and Avg Buy Price) based on transactions
    up to a specific target_date (optional).
    If target_date is None, calculates current holdings.
    """
    if df.empty:
        return pd.DataFrame(columns=['Account', 'Ticker', 'Net Quantity', 'Avg Buy Price'])

    temp_df = df.copy()

    # Ensure Date column is datetime for comparison
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])

    # Filter by date if provided
    if target_date:
        # Ensure target_date is a comparable type (Timestamp)
        target_date_ts = pd.Timestamp(target_date).normalize() # Normalize to compare date part only
        temp_df = temp_df[temp_df['Date'] <= target_date_ts]

    # Filter by account if provided
    if target_account:
        temp_df = temp_df[temp_df['Account'].str.lower() == target_account.lower()]

    if temp_df.empty:
         return pd.DataFrame(columns=['Account', 'Ticker', 'Net Quantity', 'Avg Buy Price'])

    # Calculate net quantity
    temp_df['Signed Quantity'] = temp_df.apply(lambda row: row['Quantity'] if row['Action'] == 'BUY' else -row['Quantity'], axis=1)
    holdings = temp_df.groupby(['Account', 'Ticker'])['Signed Quantity'].sum().reset_index()
    holdings = holdings.rename(columns={'Signed Quantity': 'Net Quantity'})

    # Calculate approximate average buy price
    buys = temp_df[temp_df['Action'] == 'BUY'].copy()
    buys['Cost'] = buys['Quantity'] * buys['Price']
    buy_summary = buys.groupby(['Account', 'Ticker']).agg(
        TotalQuantity=('Quantity', 'sum'),
        TotalCost=('Cost', 'sum')
    ).reset_index()

    buy_summary['Avg Buy Price'] = buy_summary.apply(
        lambda row: row['TotalCost'] / row['TotalQuantity'] if row['TotalQuantity'] != 0 else 0,
        axis=1
    )

    holdings = pd.merge(holdings, buy_summary[['Account', 'Ticker', 'Avg Buy Price']], on=['Account', 'Ticker'], how='left')
    holdings = holdings[holdings['Net Quantity'] > 0.0001] # Use tolerance for float comparison
    holdings['Avg Buy Price'] = holdings['Avg Buy Price'].fillna(0)

    # Select output columns based on whether account was specified
    if target_account:
         return holdings[['Ticker', 'Net Quantity', 'Avg Buy Price']].reset_index(drop=True)
    else:
        return holdings[['Account', 'Ticker', 'Net Quantity', 'Avg Buy Price']].reset_index(drop=True)


def display_holdings_report(df):
    """Calculates and displays the current holdings report."""
    print("\n--- Holdings Report ---")
    while True:
        account_choice = input("Enter Account Name for specific report, or leave blank for combined: ").strip()
        target_holdings_df = get_holdings(df, target_account=account_choice if account_choice else None) # Pass None for combined

        if target_holdings_df.empty:
             if account_choice:
                 print(f"No current holdings found for account '{account_choice}'.")
             else:
                 print("No current holdings found in any account.")
             return # Exit function if no holdings

        if account_choice:
            report_title = f"Holdings for Account: {account_choice}"
        else:
            report_title = "Combined Holdings Report"
        break # Exit loop once valid holdings are found


    print(f"\n{report_title}")
    print("-" * len(report_title))

    tickers = target_holdings_df['Ticker'].unique().tolist()
    if not tickers:
        print("No tickers with holdings to fetch data for.")
        return

    print("Fetching current market prices...")
    current_prices = {}

    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)

        data = yf.download(tickers,
                           start=start_date.strftime(DATE_FORMAT),
                           end=end_date.strftime(DATE_FORMAT),
                           progress=False,
                           auto_adjust=True)

        if data.empty:
             print("Warning: Could not fetch price data from yfinance in the lookback period.")
        else:
            close_prices = data['Close']
            if isinstance(close_prices, pd.Series):
                 last_valid_price = close_prices.dropna().iloc[-1] if not close_prices.dropna().empty else None
                 if last_valid_price is not None:
                     current_prices[tickers[0]] = float(last_valid_price)
            elif isinstance(close_prices, pd.DataFrame):
                 for ticker in tickers:
                     if ticker in close_prices.columns:
                         ticker_prices = close_prices[ticker].dropna()
                         if not ticker_prices.empty:
                             current_prices[ticker] = float(ticker_prices.iloc[-1])

        for ticker in tickers:
            if ticker not in current_prices:
                 current_prices[ticker] = None

        # --- Assign prices and calculate values ---
        target_holdings_df['Current Price'] = target_holdings_df['Ticker'].map(current_prices)
        target_holdings_df['Market Value'] = target_holdings_df.apply(
            lambda row: row['Net Quantity'] * row['Current Price'] if pd.notna(row['Current Price']) else None, axis=1)
        target_holdings_df['Unrealized P/L'] = target_holdings_df.apply(
            lambda row: (row['Current Price'] - row['Avg Buy Price']) * row['Net Quantity'] if pd.notna(row['Current Price']) and pd.notna(row['Avg Buy Price']) else None, axis=1)
        target_holdings_df['Unrealized P/L %'] = target_holdings_df.apply(
            lambda row: ((row['Current Price'] / row['Avg Buy Price']) - 1) * 100 if pd.notna(row['Current Price']) and pd.notna(row['Avg Buy Price']) and row['Avg Buy Price'] != 0 else None, axis=1)

        # --- Formatting for display ---
        display_df = target_holdings_df.copy()
        display_df['Net Quantity'] = display_df['Net Quantity'].map(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        display_df['Avg Buy Price'] = display_df['Avg Buy Price'].map(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        display_df['Current Price'] = display_df['Current Price'].map(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        display_df['Market Value'] = display_df['Market Value'].map(lambda x: f"{x:,.2f}" if pd.notna(x) else 'N/A')
        display_df['Unrealized P/L'] = display_df['Unrealized P/L'].map(lambda x: f"{x:,.2f}" if pd.notna(x) else 'N/A')
        display_df['Unrealized P/L %'] = display_df['Unrealized P/L %'].map(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')

        # --- Display Results ---
        if account_choice:
             print(display_df[['Ticker', 'Net Quantity', 'Avg Buy Price', 'Current Price', 'Market Value', 'Unrealized P/L', 'Unrealized P/L %']].to_string(index=False))
        else:
             print(display_df[['Account', 'Ticker', 'Net Quantity', 'Avg Buy Price', 'Current Price', 'Market Value', 'Unrealized P/L', 'Unrealized P/L %']].to_string(index=False))

        # --- Calculate and print totals ---
        market_values_num = pd.to_numeric(display_df['Market Value'].str.replace('[%,]', '', regex=True), errors='coerce')
        unrealized_pl_num = pd.to_numeric(display_df['Unrealized P/L'].str.replace('[%,]', '', regex=True), errors='coerce')
        total_market_value = market_values_num.sum()
        total_unrealized_pl = unrealized_pl_num.sum()

        print("-" * 30)
        print(f"Total Market Value: {total_market_value:,.2f}" if pd.notna(total_market_value) else "Total Market Value: N/A")
        print(f"Total Unrealized P/L: {total_unrealized_pl:,.2f}" if pd.notna(total_unrealized_pl) else "Total Unrealized P/L: N/A")
        print("-" * 30)

    except Exception as e:
        print(f"\nAn error occurred during holdings report generation: {e}")
        print("Displaying holdings without current market value:")
        if account_choice:
            print(target_holdings_df[['Ticker', 'Net Quantity', 'Avg Buy Price']].to_string(index=False))
        else:
             print(target_holdings_df[['Account', 'Ticker', 'Net Quantity', 'Avg Buy Price']].to_string(index=False))


# --- UPDATED Performance Report Function ---
def generate_performance_report(df):
    """Calculates and plots portfolio value over time."""
    print("\n--- Portfolio Performance Report ---")
    plot_individual = False # Flag for plotting individual accounts

    # 1. Determine Scope (Account or Combined)
    account_choice = input("Enter Account Name for specific report, or leave blank for combined: ").strip()
    if account_choice:
        report_scope_df = df[df['Account'].str.lower() == account_choice.lower()].copy()
        if report_scope_df.empty:
            print(f"No transactions found for account '{account_choice}'.")
            return
        report_title = f"Portfolio Performance for Account: {account_choice}"
        accounts_to_plot = [account_choice] # Only one account to plot
    else:
        report_scope_df = df.copy()
        if report_scope_df.empty:
            print("No transactions found in any account.")
            return
        report_title = "Combined Portfolio Performance"
        accounts_to_plot = report_scope_df['Account'].unique().tolist() # Get all unique accounts

        # Ask if user wants individual lines when combined
        if len(accounts_to_plot) > 1: # Only ask if there are multiple accounts
            plot_individual_choice = input("Plot individual account performance alongside the total? (y/N): ").strip().lower()
            if plot_individual_choice == 'y':
                plot_individual = True
                report_title += " (Individual Accounts Shown)"


    # Ensure Date column is datetime and normalize (start of day)
    report_scope_df['Date'] = pd.to_datetime(report_scope_df['Date']).dt.normalize()

    # 2. Determine Time Range
    if report_scope_df.empty:
        print("No transactions to analyze.")
        return
    min_date = report_scope_df['Date'].min() # Already normalized
    max_date = pd.Timestamp(datetime.now().date()) # Today normalized
    tickers = report_scope_df['Ticker'].unique().tolist()

    print(f"Analyzing performance from {min_date.strftime(DATE_FORMAT)} to {max_date.strftime(DATE_FORMAT)}...")
    print(f"Tickers involved: {', '.join(tickers)}")

    # 3. Fetch Historical Prices
    print("Fetching historical prices for all involved tickers...")
    try:
        all_prices = yf.download(tickers,
                                 start=min_date.strftime(DATE_FORMAT),
                                 end=(max_date + timedelta(days=1)).strftime(DATE_FORMAT), # +1 day buffer
                                 progress=False,
                                 auto_adjust=True)

        if all_prices.empty:
            print("Could not fetch any historical price data for the required period.")
            return

        daily_close_prices = all_prices['Close']
        if isinstance(daily_close_prices, pd.Series):
            daily_close_prices = pd.DataFrame({tickers[0]: daily_close_prices})
        # Forward-fill missing values FIRST
        daily_close_prices = daily_close_prices.ffill()
        # Backward-fill any remaining NaNs at the beginning
        daily_close_prices = daily_close_prices.bfill()

    except Exception as e:
        print(f"Error fetching historical price data: {e}")
        return

    # 4. Calculate Daily Holdings & Value
    print("Calculating daily portfolio value (this may take a moment)...")
    date_index = pd.date_range(start=min_date, end=max_date, freq='D')
    portfolio_over_time = pd.DataFrame(index=date_index)

    # Prepare transactions: sort by date, add Signed Quantity
    report_scope_df = report_scope_df.sort_values(by='Date')
    report_scope_df['Signed Quantity'] = report_scope_df.apply(
        lambda row: row['Quantity'] if row['Action'] == 'BUY' else -row['Quantity'], axis=1)

    # Initialize columns for values
    portfolio_over_time['Total Value'] = 0.0
    if plot_individual:
        for acc in accounts_to_plot:
             portfolio_over_time[f"Value_{acc}"] = 0.0

    # Iterate through each day
    for current_day in date_index:
        # Calculate holdings for ALL relevant transactions up to current_day
        relevant_transactions = report_scope_df[report_scope_df['Date'] <= current_day]
        if relevant_transactions.empty:
            current_holdings_all = pd.DataFrame(columns=['Account', 'Ticker', 'Net Quantity'])
        else:
            current_holdings_all = relevant_transactions.groupby(['Account', 'Ticker'])['Signed Quantity'].sum().reset_index()
            current_holdings_all = current_holdings_all.rename(columns={'Signed Quantity': 'Net Quantity'})
            current_holdings_all = current_holdings_all[current_holdings_all['Net Quantity'] > 0.0001]

        # Calculate values for the day
        day_combined_total = 0.0
        account_values_today = {acc: 0.0 for acc in accounts_to_plot} # Reset daily account totals

        if not current_holdings_all.empty:
            for _, holding_row in current_holdings_all.iterrows():
                account = holding_row['Account']
                ticker = holding_row['Ticker']
                quantity = holding_row['Net Quantity']
                try:
                    # Access the ffilled/bfilled price data
                    # Ensure the current_day exists in the price index
                    if current_day in daily_close_prices.index:
                         price_on_day = daily_close_prices.loc[current_day, ticker] if ticker in daily_close_prices.columns else None
                    else:
                         price_on_day = None # Date might be before price history started

                    if pd.notna(price_on_day):
                        stock_value = quantity * price_on_day
                        day_combined_total += stock_value # Add to combined total
                        if plot_individual and account in account_values_today:
                            account_values_today[account] += stock_value # Add to specific account's total
                    # else: Price is NaN (should be rare after ffill/bfill)

                except KeyError:
                     pass # Ticker column might not exist if yf download failed for it

        # Store values for the day
        portfolio_over_time.loc[current_day, 'Total Value'] = day_combined_total
        if plot_individual:
            for acc_name, acc_value in account_values_today.items():
                 portfolio_over_time.loc[current_day, f"Value_{acc_name}"] = acc_value

    # 5. **Smoothing for Plotting:** Apply ffill to the calculated values to smooth over potential zero dips IF NEEDED.
    # This is applied AFTER calculations. It assumes a zero value after a non-zero value is likely due to data gaps.
    # Apply carefully, especially if accounts genuinely start/end with zero value.
    # Let's apply only to 'Total Value' for now as the primary line.
    non_zero_found = (portfolio_over_time['Total Value'] > 0).any()
    if non_zero_found:
        portfolio_over_time['Total Value_Smoothed'] = portfolio_over_time['Total Value'].replace(0.0, np.nan).ffill().fillna(0.0)
    else: # If portfolio was always zero, keep it zero
        portfolio_over_time['Total Value_Smoothed'] = 0.0
    # Also smooth individual accounts if plotted
    if plot_individual:
        for acc in accounts_to_plot:
            col_name = f"Value_{acc}"
            smoothed_col_name = f"Value_{acc}_Smoothed"
            if col_name in portfolio_over_time.columns:
                 non_zero_acc_found = (portfolio_over_time[col_name] > 0).any()
                 if non_zero_acc_found:
                     portfolio_over_time[smoothed_col_name] = portfolio_over_time[col_name].replace(0.0, np.nan).ffill().fillna(0.0)
                 else:
                     portfolio_over_time[smoothed_col_name] = 0.0


    # 6. Optional: Save daily values to CSV
    save_csv = input("Save detailed daily portfolio values to CSV? (y/N): ").strip().lower()
    if save_csv == 'y':
        csv_filename = f"portfolio_performance_{account_choice if account_choice else 'combined'}_{min_date.strftime('%Y%m%d')}_{max_date.strftime('%Y%m%d')}.csv"
        try:
            # Save the unsmoothed data for accuracy
            portfolio_over_time.drop(columns=[col for col in portfolio_over_time if '_Smoothed' in col], errors='ignore').to_csv(csv_filename)
            print(f"Daily values saved to {csv_filename}")
        except Exception as e:
            print(f"Error saving daily values CSV: {e}")

    # 7. Plotting
    print("Generating performance chart...")
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 7)) # Wider figure

        # --- Plotting Logic ---
        # Always plot the smoothed combined total
        ax.plot(portfolio_over_time.index, portfolio_over_time['Total Value_Smoothed'], label='Combined Total', linewidth=2.5, color='black', zorder=10) # Draw total on top

        # Plot individual accounts if requested
        if plot_individual:
            # Use a cycle of colors
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            color_idx = 0
            for acc_name in accounts_to_plot:
                smoothed_col_name = f"Value_{acc_name}_Smoothed"
                if smoothed_col_name in portfolio_over_time.columns:
                     # Check if account ever had non-zero value before plotting
                     if (portfolio_over_time[smoothed_col_name] > 0.001).any():
                         ax.plot(portfolio_over_time.index,
                                 portfolio_over_time[smoothed_col_name],
                                 label=f'Account: {acc_name}',
                                 alpha=0.8,
                                 color=colors[color_idx % len(colors)]) # Cycle through colors
                         color_idx += 1
                     else:
                          print(f"Skipping plot for Account: {acc_name} (value was always zero or near-zero).")
        # --- End Plotting Logic ---

        # Formatting the plot
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (INR)") # Assuming INR currency
        ax.set_title(report_title)
        ax.legend()
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3) # Minor grid

        # Improve date formatting on x-axis
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate() # Auto rotate dates

        # Format Y axis to show currency nicely potentially
        # from matplotlib.ticker import FuncFormatter
        # ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₹{x:,.0f}')) # Example for INR

        plt.tight_layout() # Adjust layout
        plt.savefig('output.png') # Display the plot

    except Exception as e:
        print(f"Error generating plot: {e}")
        # Optional: print unsmoothed data as fallback
        # print("Displaying calculated daily values (unsmoothed):")
        # print(portfolio_over_time.drop(columns=[col for col in portfolio_over_time if '_Smoothed' in col], errors='ignore').head())


def view_transactions(df):
    """Displays transactions, optionally filtered by account."""
    print("\n--- View Transactions ---")
    if df.empty:
        print("No transactions recorded yet.")
        return

    account_choice = input("Enter Account Name to filter, or leave blank for all: ").strip()

    # Ensure Date column is datetime for sorting, then format for display
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
    df_display = df.copy()
    if not df_display.empty:
        df_display['Date'] = df_display['Date'].dt.strftime(DATE_FORMAT) # Format for display


    if account_choice:
        filtered_df = df_display[df_display['Account'].str.lower() == account_choice.lower()]
        if filtered_df.empty:
            print(f"No transactions found for account '{account_choice}'.")
            return
        print(f"\n--- Transactions for Account: {account_choice} ---")
        print(filtered_df.sort_values(by=['Date', 'Ticker']).to_string(index=False))
    else:
        print("\n--- All Transactions ---")
        if not df_display.empty:
            print(df_display.sort_values(by=['Date', 'Account', 'Ticker']).to_string(index=False))
        else:
            print("No transactions to display.")


# --- Main Application Loop (No changes needed here) ---
def main():
    transactions_df = load_transactions()

    while True:
        print("\n===== Portfolio Tracker Menu =====")
        print("1. Add Transaction (Buy/Sell)")
        print("2. View Transactions")
        print("3. Generate Holdings Report (Current Snapshot)")
        print("4. Generate Portfolio Performance Report (Chart)")
        print("5. Save Transactions")
        print("0. Save and Exit")
        print("================================")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            transactions_df['Date'] = pd.to_datetime(transactions_df['Date']) # Ensure datetime before adding
            transactions_df = add_transaction(transactions_df)
        elif choice == '2':
            view_transactions(transactions_df)
        elif choice == '3':
            display_holdings_report(transactions_df)
        elif choice == '4':
            generate_performance_report(transactions_df)
        elif choice == '5':
            save_transactions(transactions_df)
        elif choice == '0':
            save_transactions(transactions_df)
            print("Exiting Portfolio Tracker. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

# --- Run the Application ---
if __name__ == "__main__":
    main()