import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings

# Suppress specific pandas warnings (optional)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# --- Configuration ---
DATA_FILE = 'portfolio_transactions.csv'
DATE_FORMAT = '%Y-%m-%d' # Consistent date format

# --- File Handling ---
def load_transactions(filename=DATA_FILE):
    """Loads transactions from the CSV file into a pandas DataFrame."""
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, parse_dates=['Date'])
            # Ensure correct data types after loading
            df['Date'] = pd.to_datetime(df['Date'])
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
        df.to_csv(filename, index=False, date_format=DATE_FORMAT)
        print(f"Transactions saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving transactions to {filename}: {e}")

# --- Core Logic ---
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
                trans_date = datetime.now().date()
            else:
                trans_date = datetime.strptime(date_str, DATE_FORMAT).date()
            break
        except ValueError:
            print(f"Invalid date format. Please use {DATE_FORMAT.replace('%', '').upper()}.")

    new_transaction = pd.DataFrame([{
        'Account': account,
        'Ticker': ticker,
        'Action': action,
        'Quantity': quantity,
        'Price': price,
        'Date': pd.to_datetime(trans_date) # Ensure Timestamp object
    }])

    df = pd.concat([df, new_transaction], ignore_index=True)
    print("Transaction added successfully.")
    return df

def get_holdings(df, target_account=None):
    """Calculates current holdings based on transactions."""
    if df.empty:
        return pd.DataFrame(columns=['Account', 'Ticker', 'Net Quantity', 'Avg Buy Price'])

    temp_df = df.copy()
    if target_account:
        temp_df = temp_df[temp_df['Account'].str.lower() == target_account.lower()]

    if temp_df.empty:
         return pd.DataFrame(columns=['Account', 'Ticker', 'Net Quantity', 'Avg Buy Price'])

    # Calculate net quantity
    temp_df['Signed Quantity'] = temp_df.apply(lambda row: row['Quantity'] if row['Action'] == 'BUY' else -row['Quantity'], axis=1)
    holdings = temp_df.groupby(['Account', 'Ticker'])['Signed Quantity'].sum().reset_index()
    holdings = holdings.rename(columns={'Signed Quantity': 'Net Quantity'})

    # Calculate approximate average buy price (simple version: total cost of buys / total quantity bought)
    # More accurate methods (FIFO, LIFO, Weighted Avg) are complex and depend on sell order matching.
    buys = temp_df[temp_df['Action'] == 'BUY'].copy()
    buys['Cost'] = buys['Quantity'] * buys['Price']
    buy_summary = buys.groupby(['Account', 'Ticker']).agg(
        TotalQuantity=('Quantity', 'sum'),
        TotalCost=('Cost', 'sum')
    ).reset_index()
    buy_summary['Avg Buy Price'] = buy_summary['TotalCost'] / buy_summary['TotalQuantity']

    # Merge holdings with average buy price
    holdings = pd.merge(holdings, buy_summary[['Account', 'Ticker', 'Avg Buy Price']], on=['Account', 'Ticker'], how='left')

    # Filter out stocks that have been completely sold (Net Quantity <= 0)
    holdings = holdings[holdings['Net Quantity'] > 0.0001] # Use tolerance for float comparison
    holdings['Avg Buy Price'] = holdings['Avg Buy Price'].fillna(0) # Handle cases if only sells exist somehow

    return holdings[['Account', 'Ticker', 'Net Quantity', 'Avg Buy Price']].reset_index(drop=True)


def display_holdings_report(df):
    """Calculates and displays the current holdings report."""
    print("\n--- Holdings Report ---")
    while True:
        account_choice = input("Enter Account Name for specific report, or leave blank for combined: ").strip()
        if account_choice:
            holdings_df = get_holdings(df, target_account=account_choice)
            if holdings_df.empty:
                 print(f"No current holdings found for account '{account_choice}'.")
                 return
            report_title = f"Holdings for Account: {account_choice}"
            break
        else:
            holdings_df = get_holdings(df)
            if holdings_df.empty:
                 print("No current holdings found in any account.")
                 return
            report_title = "Combined Holdings Report"
            break

    print(f"\n{report_title}")
    print("-" * len(report_title))

    tickers = holdings_df['Ticker'].unique().tolist()
    if not tickers:
        print("No tickers to fetch data for.") # Should not happen if holdings_df is not empty, but good check
        if not holdings_df.empty:
             print(holdings_df[['Account', 'Ticker', 'Net Quantity', 'Avg Buy Price']].to_string(index=False))
        return

    print("Fetching current market prices...")
    current_prices = {} # Dictionary to store fetched prices {ticker: price}

    try:
        # Fetch data for all unique tickers at once
        # Use auto_adjust=True for simplicity (handles splits/dividends in price)
        # Look back a few days to ensure we get the latest closing price
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7) # Look back 7 days

        # Suppress yfinance progress messages if desired
        data = yf.download(tickers,
                           start=start_date.strftime(DATE_FORMAT),
                           end=end_date.strftime(DATE_FORMAT),
                           progress=False,
                           auto_adjust=True) # auto_adjust=True recommended

        if data.empty:
             print("Warning: Could not fetch price data from yfinance in the lookback period.")
        else:
            # Get the 'Close' prices
            close_prices = data['Close']

            # Find the last valid price for each ticker in the fetched data
            if isinstance(close_prices, pd.Series): # Case: Only one ticker fetched
                 # Find last non-NaN value in the Series
                 last_valid_price = close_prices.dropna().iloc[-1] if not close_prices.dropna().empty else None
                 if last_valid_price is not None:
                     current_prices[tickers[0]] = float(last_valid_price)

            elif isinstance(close_prices, pd.DataFrame): # Case: Multiple tickers fetched
                 for ticker in tickers:
                     if ticker in close_prices.columns:
                         # Find last non-NaN value for this specific ticker's column
                         ticker_prices = close_prices[ticker].dropna()
                         if not ticker_prices.empty:
                             current_prices[ticker] = float(ticker_prices.iloc[-1])
                         else:
                             print(f"Warning: No valid closing price found for {ticker} in the lookback period.")
                     else:
                         print(f"Warning: Ticker {ticker} column not found in fetched data.")
            else:
                 print("Warning: Unexpected data structure received from yfinance.")

        # Ensure all tickers in holdings have an entry in current_prices, even if it's None
        for ticker in tickers:
            if ticker not in current_prices:
                 current_prices[ticker] = None # Mark as not found
                 print(f"Warning: Could not determine current price for {ticker}.")

        # --- Assign prices and calculate values ---
        holdings_df['Current Price'] = holdings_df['Ticker'].map(current_prices)

        # Calculate Market Value only if Current Price is available (not None)
        holdings_df['Market Value'] = holdings_df.apply(
            lambda row: row['Net Quantity'] * row['Current Price'] if pd.notna(row['Current Price']) else None,
            axis=1
        )
        # Calculate P/L only if Current Price and Avg Buy Price are valid
        holdings_df['Unrealized P/L'] = holdings_df.apply(
            lambda row: (row['Current Price'] - row['Avg Buy Price']) * row['Net Quantity'] if pd.notna(row['Current Price']) and pd.notna(row['Avg Buy Price']) else None,
            axis=1
        )
        # Calculate P/L % only if prices are valid and Avg Buy Price is not zero
        holdings_df['Unrealized P/L %'] = holdings_df.apply(
            lambda row: ((row['Current Price'] / row['Avg Buy Price']) - 1) * 100 if pd.notna(row['Current Price']) and pd.notna(row['Avg Buy Price']) and row['Avg Buy Price'] != 0 else None,
            axis=1
        )

        # --- Formatting for display (handle potential None values gracefully) ---
        # Use .map() with a lambda function to apply formatting only to valid numbers
        holdings_df['Net Quantity'] = holdings_df['Net Quantity'].map(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        holdings_df['Avg Buy Price'] = holdings_df['Avg Buy Price'].map(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        holdings_df['Current Price'] = holdings_df['Current Price'].map(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        holdings_df['Market Value'] = holdings_df['Market Value'].map(lambda x: f"{x:,.2f}" if pd.notna(x) else 'N/A') # Added comma separator
        holdings_df['Unrealized P/L'] = holdings_df['Unrealized P/L'].map(lambda x: f"{x:,.2f}" if pd.notna(x) else 'N/A') # Added comma separator
        holdings_df['Unrealized P/L %'] = holdings_df['Unrealized P/L %'].map(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')

        # --- Display Results ---
        print(holdings_df.to_string(index=False))

        # --- Calculate and print totals (only sum valid numbers) ---
        # Convert back to numeric safely for summation, coercing errors to NaN
        market_values_num = pd.to_numeric(holdings_df['Market Value'].str.replace('[%,]', '', regex=True), errors='coerce')
        unrealized_pl_num = pd.to_numeric(holdings_df['Unrealized P/L'].str.replace('[%,]', '', regex=True), errors='coerce')

        total_market_value = market_values_num.sum()
        total_unrealized_pl = unrealized_pl_num.sum()

        print("-" * 30)
        # Print totals only if they are valid numbers
        print(f"Total Market Value: {total_market_value:,.2f}" if pd.notna(total_market_value) else "Total Market Value: N/A")
        print(f"Total Unrealized P/L: {total_unrealized_pl:,.2f}" if pd.notna(total_unrealized_pl) else "Total Unrealized P/L: N/A")
        print("-" * 30)


    except Exception as e:
        print(f"\nAn error occurred during holdings report generation: {e}")
        print("Displaying holdings without current market value:")
        # Select only core columns if calculations failed
        print(holdings_df[['Account', 'Ticker', 'Net Quantity', 'Avg Buy Price']].to_string(index=False))


def display_historical_price_report(df):
    """Fetches and displays historical closing prices for a specific holding."""
    print("\n--- Historical Price Report ---")

    # First, show current holdings to help user choose
    all_holdings = get_holdings(df)
    if all_holdings.empty:
        print("No current holdings available to generate historical report for.")
        return

    print("Current Holdings:")
    print(all_holdings[['Account', 'Ticker', 'Net Quantity']].to_string(index=False))
    print("-" * 20)

    # Get user input for which holding
    while True:
        try:
            choice = int(input("Enter the row number (starting from 0) of the holding you want history for: "))
            if 0 <= choice < len(all_holdings):
                selected_holding = all_holdings.iloc[choice]
                break
            else:
                print("Invalid row number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except IndexError:
             print("Invalid row number.")

    account = selected_holding['Account']
    ticker = selected_holding['Ticker']

    # Find the first buy date for this specific holding in the account
    account_ticker_buys = df[(df['Account'] == account) & (df['Ticker'] == ticker) & (df['Action'] == 'BUY')]
    if account_ticker_buys.empty:
        print(f"Could not find buy transactions for {ticker} in account {account}.")
        # Fallback: Find the earliest transaction date for this ticker overall
        first_date = df[df['Ticker'] == ticker]['Date'].min()
        if pd.isna(first_date):
             print("Cannot determine a start date.")
             return
    else:
         first_date = account_ticker_buys['Date'].min()


    print(f"\nFetching historical data for {ticker} (Account: {account}) from {first_date.strftime(DATE_FORMAT)} onwards.")

    # Get end date from user
    while True:
        end_date_str = input(f"Enter End Date ({DATE_FORMAT.replace('%', '').upper()}), leave blank for today: ").strip()
        try:
            if not end_date_str:
                end_date = datetime.now().date()
            else:
                end_date = datetime.strptime(end_date_str, DATE_FORMAT).date()

            if end_date >= first_date.date(): # Compare dates only
                 break
            else:
                 print("End date cannot be before the first buy date.")

        except ValueError:
            print(f"Invalid date format. Please use {DATE_FORMAT.replace('%', '').upper()}.")

    # Fetch historical data using yfinance
    try:
        print(f"Fetching data from {first_date.strftime(DATE_FORMAT)} to {end_date.strftime(DATE_FORMAT)}...")
        history = yf.download(ticker,
                              start=first_date.strftime(DATE_FORMAT),
                              end=(end_date + timedelta(days=1)).strftime(DATE_FORMAT), # yf end is exclusive
                              progress=False)

        if history.empty:
            print(f"No historical data found for {ticker} in the specified date range.")
            return

        print(f"\n--- Closing Prices for {ticker} ({account}) ---")
        print(f"--- From {first_date.strftime(DATE_FORMAT)} to {end_date.strftime(DATE_FORMAT)} ---")

        # Select and format relevant columns
        history_close = history[['Close']].round(2)
        history_close.index.name = 'Date' # Rename index
        print(history_close)

    except Exception as e:
        print(f"An error occurred while fetching historical data: {e}")


def view_transactions(df):
    """Displays transactions, optionally filtered by account."""
    print("\n--- View Transactions ---")
    if df.empty:
        print("No transactions recorded yet.")
        return

    account_choice = input("Enter Account Name to filter, or leave blank for all: ").strip()

    if account_choice:
        filtered_df = df[df['Account'].str.lower() == account_choice.lower()]
        if filtered_df.empty:
            print(f"No transactions found for account '{account_choice}'.")
            return
        print(f"\n--- Transactions for Account: {account_choice} ---")
        print(filtered_df.to_string(index=False))
    else:
        print("\n--- All Transactions ---")
        print(df.sort_values(by=['Date', 'Account', 'Ticker']).to_string(index=False))


# --- Main Application Loop ---
def main():
    transactions_df = load_transactions()

    while True:
        print("\n===== Portfolio Tracker Menu =====")
        print("1. Add Transaction (Buy/Sell)")
        print("2. View Transactions")
        print("3. Generate Holdings Report (Current Snapshot)")
        print("4. Generate Historical Price Report (for a holding)")
        print("5. Save Transactions")
        print("0. Save and Exit")
        print("================================")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            transactions_df = add_transaction(transactions_df)
        elif choice == '2':
            view_transactions(transactions_df)
        elif choice == '3':
            display_holdings_report(transactions_df)
        elif choice == '4':
            display_historical_price_report(transactions_df)
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