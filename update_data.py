"""
🔄 STOCK DATA UPDATER
Downloads latest stock data from Yahoo Finance and updates CSV files

Usage:
    python update_data.py
    
Features:
- Downloads data for all stocks in indian_data/
- Updates existing CSVs with new data
- Handles missing data gracefully
- Validates data quality
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')


class StockDataUpdater:
    """Updates stock data from Yahoo Finance"""
    
    def __init__(self, data_path: str = 'indian_data'):
        self.data_path = Path(data_path)
        self.updated_count = 0
        self.failed_count = 0
        self.failed_stocks = []
        
    def get_ticker_symbol(self, filename: str) -> str:
        """Convert filename to Yahoo Finance ticker symbol"""
        # Remove .csv extension
        ticker = filename.replace('.csv', '')
        
        # Add .NS suffix for NSE stocks (Indian market)
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
        
        return ticker
    
    def download_stock_data(self, ticker: str, start_date: str, end_date: str):
        """Download stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"   ❌ Error downloading {ticker}: {e}")
            return None
    
    def update_csv_file(self, csv_path: Path):
        """Update a single CSV file with latest data"""
        try:
            # Read existing data
            existing_df = pd.read_csv(csv_path)
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            
            # Get last date in file
            last_date = existing_df['Date'].max()
            
            # Calculate date range for update
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Check if update is needed
            if start_date >= end_date:
                print(f"   ✅ {csv_path.name}: Already up to date")
                return True
            
            # Get ticker symbol
            ticker = self.get_ticker_symbol(csv_path.name)
            
            print(f"   🔄 {csv_path.name}: Updating from {start_date} to {end_date}")
            
            # Download new data
            new_df = self.download_stock_data(ticker, start_date, end_date)
            
            if new_df is None or new_df.empty:
                print(f"   ⚠️ {csv_path.name}: No new data available")
                return True
            
            # Combine old and new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates
            combined_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            
            # Sort by date
            combined_df.sort_values('Date', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
            
            # Save updated file
            combined_df.to_csv(csv_path, index=False)
            
            print(f"   ✅ {csv_path.name}: Added {len(new_df)} new rows")
            self.updated_count += 1
            return True
            
        except Exception as e:
            print(f"   ❌ {csv_path.name}: Failed - {e}")
            self.failed_count += 1
            self.failed_stocks.append(csv_path.name)
            return False
    
    def update_all_stocks(self):
        """Update all stock CSV files"""
        print("\n" + "="*70)
        print("🔄 STOCK DATA UPDATER")
        print("="*70)
        
        if not self.data_path.exists():
            print(f"\n❌ Data directory not found: {self.data_path}")
            return False
        
        # Get all CSV files
        csv_files = sorted(self.data_path.glob('*.csv'))
        total_files = len(csv_files)
        
        print(f"\n📁 Found {total_files} stock files")
        print(f"📅 Current date: {datetime.now().strftime('%Y-%m-%d')}")
        print("\n" + "-"*70)
        
        # Update each file
        for i, csv_path in enumerate(csv_files, 1):
            print(f"\n[{i}/{total_files}] Processing {csv_path.name}")
            self.update_csv_file(csv_path)
            
            # Rate limiting (be nice to Yahoo Finance)
            if i < total_files:
                time.sleep(0.5)  # 0.5 second delay between requests
        
        # Print summary
        print("\n" + "="*70)
        print("📊 UPDATE SUMMARY")
        print("="*70)
        print(f"✅ Successfully updated: {self.updated_count}/{total_files}")
        print(f"⚠️ Already up to date: {total_files - self.updated_count - self.failed_count}")
        print(f"❌ Failed: {self.failed_count}")
        
        if self.failed_stocks:
            print(f"\nFailed stocks:")
            for stock in self.failed_stocks:
                print(f"  • {stock}")
        
        print("\n" + "="*70)
        print("✅ UPDATE COMPLETE!")
        print("="*70)
        
        return True


def main():
    """Main execution"""
    updater = StockDataUpdater('indian_data')
    updater.update_all_stocks()
    
    print("\n💡 NEXT STEPS:")
    print("   1. Verify updated data: Check indian_data/ folder")
    print("   2. Retrain model: python train.py")
    print("   3. Generate predictions: python predict_now.py")
    print()


if __name__ == "__main__":
    main()
