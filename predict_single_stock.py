"""
🔮 SINGLE STOCK PREDICTOR
Get prediction for any stock from pre-generated CSV files
"""

import pandas as pd
from datetime import datetime
import os
import glob


class SingleStockPredictor:
    def __init__(self, predictions_folder='predictions'):
        """Initialize predictor with CSV predictions"""
        self.predictions_folder = predictions_folder
        
        print("🔮 Loading Stock Predictions...")
        print("="*60)
        
        # Find latest prediction file
        csv_files = glob.glob(f"{predictions_folder}/predictions_*.csv")
        
        if not csv_files:
            print(f"❌ ERROR: No prediction files found in '{predictions_folder}/'")
            print("💡 Run 'python predict_now.py' first to generate predictions")
            raise FileNotFoundError(f"No prediction files in {predictions_folder}/")
        
        # Get most recent file
        self.latest_file = max(csv_files, key=os.path.getctime)
        
        # Extract date from filename
        filename = os.path.basename(self.latest_file)
        self.prediction_date = filename.replace('predictions_', '').replace('.csv', '')
        
        # Load predictions
        self.predictions_df = pd.read_csv(self.latest_file)
        
        print(f"✅ Loaded predictions from: {filename}")
        print(f"📅 Prediction date: {self.prediction_date}")
        print(f"📊 Total stocks: {len(self.predictions_df)}")
        print("="*60)
        
    def predict_stock(self, ticker):
        """Get prediction for a single stock"""
        
        # Validate ticker
        ticker = ticker.upper()
        stock_data = self.predictions_df[self.predictions_df['Ticker'] == ticker]
        
        if stock_data.empty:
            print(f"\n❌ ERROR: Ticker '{ticker}' not found in predictions!")
            print(f"\n💡 Did you mean one of these?")
            
            # Find similar tickers
            all_tickers = self.predictions_df['Ticker'].tolist()
            similar = [t for t in all_tickers if ticker in t]
            
            if similar:
                print(f"   {', '.join(similar[:5])}")
            else:
                print(f"   Type 'list' to see all available stocks")
            
            return None
        
        # Extract stock data
        stock = stock_data.iloc[0]
        
        return {
            'ticker': stock['Ticker'],
            'sector': stock['Sector'],
            'direction': stock['Direction'],
            'confidence': stock['Confidence_%'],
            'up_probability': stock['UP_Probability'] * 100,
            'expected_return': stock['Expected_Return_%'],
            'ranking_score': stock['Ranking_Score'],
            'rank': stock['Rank'],
            'total_stocks': len(self.predictions_df)
        }
    
    def display_prediction(self, result):
        """Display prediction in a nice format"""
        if result is None:
            return
        
        print("\n" + "="*60)
        print("🎯 STOCK PREDICTION")
        print("="*60)
        print(f"📊 Stock: {result['ticker']}")
        print(f"🏢 Sector: {result['sector']}")
        print(f"📅 Prediction for: Next trading day after {self.prediction_date}")
        print("-"*60)
        
        # Direction with emoji
        if result['direction'] == "UP":
            direction_emoji = "⬆️ 📈"
            direction_color = "🟢"
        else:
            direction_emoji = "⬇️ 📉"
            direction_color = "🔴"
        
        print(f"{direction_color} Direction: {result['direction']} {direction_emoji}")
        print(f"💪 Confidence: {result['confidence']:.1f}%")
        print(f"📊 UP Probability: {result['up_probability']:.1f}%")
        print(f"💰 Expected Return: {result['expected_return']:.2f}% (weekly)")
        print(f"🏆 Rank: #{result['rank']} out of {result['total_stocks']} stocks")
        print(f"⭐ Ranking Score: {result['ranking_score']:.4f}")
        print("="*60)
        
        # Interpretation
        print("\n💡 INTERPRETATION:")
        
        # Confidence strength
        if result['confidence'] > 50:
            strength = "🔥 VERY HIGH"
        elif result['confidence'] > 35:
            strength = "✅ HIGH"
        elif result['confidence'] > 20:
            strength = "⚠️ MODERATE"
        else:
            strength = "❌ LOW"
        
        print(f"   Confidence Level: {strength}")
        
        # Quality based on rank
        if result['rank'] <= 5:
            quality = "🏆 EXCELLENT (Top 5!)"
        elif result['rank'] <= 10:
            quality = "🌟 EXCELLENT (Top 10!)"
        elif result['rank'] <= 20:
            quality = "✅ VERY GOOD (Top 20)"
        elif result['rank'] <= 50:
            quality = "👍 GOOD (Top 50)"
        else:
            quality = "📊 AVERAGE"
        
        print(f"   Quality Rank: {quality}")
        
        # Trading suggestion
        print("\n📈 TRADING SUGGESTION:")
        if result['direction'] == "UP" and result['confidence'] > 35 and result['rank'] <= 20:
            print("   ✅ STRONG BUY - High confidence + Top 20 rank")
            print("   💰 Suggested allocation: 10-15% of portfolio")
        elif result['direction'] == "UP" and result['confidence'] > 20:
            print("   ✅ BUY - Moderate confidence upward prediction")
            print("   💰 Suggested allocation: 5-10% of portfolio")
        elif result['direction'] == "UP":
            print("   ⚠️ WEAK BUY - Low confidence, higher risk")
            print("   💰 Suggested allocation: 2-5% of portfolio")
        elif result['direction'] == "DOWN" and result['confidence'] > 35:
            print("   ❌ STRONG SELL/AVOID - High confidence downward prediction")
            print("   🚫 Do NOT buy this stock")
        elif result['direction'] == "DOWN" and result['confidence'] > 20:
            print("   ❌ SELL/AVOID - Moderate confidence downward prediction")
            print("   🚫 Avoid or reduce position")
        else:
            print("   ⚠️ NEUTRAL - Low confidence, uncertain direction")
            print("   🤔 Wait for better signal")
        
        # Risk assessment
        print("\n⚠️ RISK ASSESSMENT:")
        if result['confidence'] > 40:
            risk = "LOW - High conviction"
        elif result['confidence'] > 25:
            risk = "MODERATE - Reasonable confidence"
        else:
            risk = "HIGH - Low confidence signal"
        print(f"   Risk Level: {risk}")
        
        # Investment calculation
        if result['direction'] == "UP" and result['rank'] <= 50:
            print("\n💰 INVESTMENT CALCULATOR:")
            for amount in [10000, 50000, 100000, 200000]:
                expected_profit = amount * (result['expected_return'] / 100)
                print(f"   ₹{amount:,} → Expected profit: ₹{expected_profit:,.0f} (weekly)")
        
        print("="*60 + "\n")
    
    def list_stocks(self):
        """List all available stocks"""
        print("\n📋 AVAILABLE STOCKS:")
        print("="*60)
        
        tickers = sorted(self.predictions_df['Ticker'].tolist())
        
        for i in range(0, len(tickers), 6):
            row = tickers[i:i+6]
            print("  " + ", ".join(f"{t:<12}" for t in row))
        
        print("="*60)
        print(f"Total: {len(tickers)} stocks")
        print("="*60 + "\n")
    
    def show_top_picks(self, n=10):
        """Show top N stock picks"""
        print(f"\n🏆 TOP {n} STOCK PICKS:")
        print("="*60)
        
        top_stocks = self.predictions_df.head(n)
        
        for idx, row in top_stocks.iterrows():
            direction_emoji = "⬆️" if row['Direction'] == "UP" else "⬇️"
            print(f"#{row['Rank']:2d}. {row['Ticker']:<12} {direction_emoji} {row['Direction']:<5} "
                  f"Conf: {row['Confidence_%']:5.1f}%  Return: {row['Expected_Return_%']:6.2f}%  "
                  f"Sector: {row['Sector']}")
        
        print("="*60 + "\n")


def main():
    """Main interactive loop"""
    
    print("\n" + "="*60)
    print("🎯 FINGAT SINGLE STOCK PREDICTION TOOL")
    print("="*60)
    print("📊 Get AI predictions for any stock")
    print("🤖 Powered by Graph Neural Networks (GATv2)")
    print("="*60 + "\n")
    
    try:
        predictor = SingleStockPredictor('predictions')
    except Exception as e:
        print(f"❌ Error loading predictor: {e}")
        return
    
    print("\n💡 COMMANDS:")
    print("   • Type stock ticker (e.g., INFY, TCS, RELIANCE)")
    print("   • Type 'list' to see all available stocks")
    print("   • Type 'top' or 'top10' to see top picks")
    print("   • Type 'exit' or 'quit' to close")
    print("-"*60 + "\n")
    
    while True:
        try:
            # Get user input
            command = input("🔍 Enter command or ticker: ").strip()
            
            # Check for exit
            if command.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye! Happy trading! 🚀\n")
                break
            
            # Skip empty input
            if not command:
                continue
            
            # Check for list command
            if command.lower() == 'list':
                predictor.list_stocks()
                continue
            
            # Check for top command
            if command.lower() in ['top', 'top5', 'top10', 'top20']:
                n = 10  # default
                if 'top5' in command.lower():
                    n = 5
                elif 'top20' in command.lower():
                    n = 20
                predictor.show_top_picks(n)
                continue
            
            # Assume it's a ticker
            result = predictor.predict_stock(command)
            
            # Display result
            predictor.display_prediction(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy trading! 🚀\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
