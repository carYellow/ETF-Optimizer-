import yfinance as yf
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import os

class StockDataLoader:
    def __init__(self, start_date: str = "2010-01-01"):
        """
        Initialize the StockDataLoader.
        
        Args:
            start_date (str): Start date for historical data in YYYY-MM-DD format
        """
        self.start_date = start_date
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
    def get_sp500_symbols(self) -> List[str]:
        """
        Get list of S&P 500 symbols.
        
        Returns:
            List[str]: List of stock symbols
        """
        # Check if symbols are already saved locally
        symbols_file = "data/raw/sp500_symbols.csv"
        if os.path.exists(symbols_file):
            print(f"Loading S&P 500 symbols from local file: {symbols_file}")
            return pd.read_csv(symbols_file)['Symbol'].tolist()
            
        print("Fetching S&P 500 symbols from Wikipedia...")
        # If not found locally, fetch from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        
        # Save to local file
        os.makedirs(os.path.dirname(symbols_file), exist_ok=True)
        df.to_csv(symbols_file, index=False)
        print(f"Saved S&P 500 symbols to local file: {symbols_file}")
        
        return df['Symbol'].tolist()
    
    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch historical data for a given stock symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        # Check if data is already saved locally
        data_file = f"data/raw/{symbol}_data.csv"
        if os.path.exists(data_file):
            print(f"Loading stock data for {symbol} from local file: {data_file}")
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            df['Symbol'] = symbol
            return df
            
        try:
            print(f"Fetching stock data for {symbol} from yfinance...")
            # If not found locally, fetch from yfinance
            stock = yf.Ticker(symbol)
            df = stock.history(start=self.start_date, end=self.end_date)
            df['Symbol'] = symbol
            
            # Save to local file
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            df.to_csv(data_file)
            print(f"Saved stock data for {symbol} to local file: {data_file}")
            
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_sp500_index(self) -> pd.DataFrame:
        """
        Fetch S&P 500 index data.
        
        Returns:
            pd.DataFrame: Historical OHLCV data for S&P 500
        """
        return self.fetch_stock_data("^GSPC")
    
    def prepare_training_data(self, 
                            symbols: Optional[List[str]] = None,
                            min_data_points: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data for multiple stocks.
        
        Args:
            symbols (List[str], optional): List of stock symbols. If None, uses all S&P 500 stocks
            min_data_points (int): Minimum number of data points required for a stock
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Stock data and S&P 500 index data
        """
        if symbols is None:
            symbols = self.get_sp500_symbols()
        
        print("Preparing training data...")
        # Fetch S&P 500 index data
        sp500_data = self.fetch_sp500_index()
        
        # Fetch individual stock data
        stock_data_list = []
        for symbol in symbols:
            df = self.fetch_stock_data(symbol)
            if len(df) >= min_data_points:
                stock_data_list.append(df)
        
        # Combine all stock data
        stock_data = pd.concat(stock_data_list, axis=0)
        print(f"Prepared training data with {len(stock_data_list)} stocks")
        
        return stock_data, sp500_data
    
    def calculate_returns(self, 
                         stock_data: pd.DataFrame,
                         sp500_data: pd.DataFrame,
                         window: int = 5) -> pd.DataFrame:
        """
        Calculate returns and create labels for model training.
        
        Args:
            stock_data (pd.DataFrame): Stock data
            sp500_data (pd.DataFrame): S&P 500 index data
            window (int): Number of days for return calculation
            
        Returns:
            pd.DataFrame: Data with calculated returns and labels
        """
        print(f"Calculating returns with {window}-day window...")
        # Calculate returns for stocks
        stock_data['Returns'] = stock_data.groupby('Symbol')['Close'].pct_change(window)
        
        # Calculate returns for S&P 500
        sp500_data['Returns'] = sp500_data['Close'].pct_change(window)
        
        # Merge stock data with S&P 500 data
        merged_data = pd.merge(
            stock_data,
            sp500_data[['Returns']],
            left_index=True,
            right_index=True,
            suffixes=('', '_SP500')
        )
        
        # Create binary labels (1 if stock outperforms S&P 500, 0 otherwise)
        merged_data['Label'] = (merged_data['Returns'] > merged_data['Returns_SP500']).astype(int)
        print("Finished calculating returns and creating labels")
        
        return merged_data 